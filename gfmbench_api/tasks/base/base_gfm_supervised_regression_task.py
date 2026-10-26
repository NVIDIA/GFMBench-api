# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmbench_api.metrics import RegressionPearsonR, RegressionR2
from gfmbench_api.tasks.base.base_gfm_supervised_task import BaseGFMSupervisedTask


class BaseGFMSupervisedRegressionTask(BaseGFMSupervisedTask):
    """
    Base class for sequence-level and binned regression tasks (e.g. CAGE).

    Dataset format: (sequence, label, conditional_input) tuples
    Model inference: infer_sequence_to_regression_outputs(sequences, conditional_input),
    returning the final predictions of the model's regression head. It is not part of
    BaseGFMModel; a model (e.g. a fine-tuning wrapper) that owns a regression head implements
    it and is responsible for any per-token head application and pooling into bins (see
    gfmbench_api.utils.binning.pool_tokens_to_bins and sequence_pos_to_prob_pos).
    The model is queried through the safe model call, as in classification: if it does not
    implement the method or returns None, the task's metrics are None. Outputs whose width
    differs from num_labels raise ValueError.

    Not supported by this contract: heads applied after pooling (e.g. an MLP head on pooled
    bin embeddings), pooling by bp overlap, and full fine-tuning of the backbone. The
    reference wrapper applies the head per token before pooling (equivalent to pooling first
    only for a linear head); its valid-token count is approximate for tokenizers whose last
    nucleotide does not map to the last output position, and empty bins are 0 rather than
    the head's bias.

    The bin size determines the label and prediction layout:
        - get_bin_size_bp() is None: sequence-level, [batch_size, num_labels] -
          one value per label per sequence (reported as num_bins = 1)
        - get_bin_size_bp() is an int: binned, [batch_size, num_bins, num_labels] -
          one value per label per bin, where
          num_bins = max_sequence_length // bin_size_bp (must divide evenly and
          be at least 2)

    Evaluation reports macro Pearson r and macro R^2, computed per label after pooling
    all leading dimensions (samples, and bins for binned tasks).

    Subclasses must implement:
        - get_bin_size_bp(): Return the bin size in bp for binned outputs, or None
          for sequence-level outputs
        - _get_num_labels(): Return number of regression outputs per sequence or bin
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for regression tasks."""
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "num_labels": self._validate_num_labels(),
            "task_type": "regression",
            "classification_mode": "single_label",
            "is_variant_effect_prediction": False,
            "num_bins": self._get_num_bins(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _get_num_bins(self) -> int:
        """Return the number of output bins per sequence (1 for sequence-level tasks).

        Derived as ``max_sequence_length // get_bin_size_bp()``. Raises if the bin
        size is invalid, the sequence length is not an exact multiple of it, or the
        result is fewer than 2 bins (a single bin is not a binned output).
        """
        bin_size_bp = self.get_bin_size_bp()
        if bin_size_bp is None:
            return 1
        if (
            isinstance(bin_size_bp, bool)
            or not isinstance(bin_size_bp, int)
            or bin_size_bp <= 0
        ):
            raise ValueError(
                f"get_bin_size_bp() must return a positive int or None, got {bin_size_bp!r}"
            )
        if self.max_sequence_length % bin_size_bp != 0:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be a multiple of "
                f"bin_size_bp ({bin_size_bp})"
            )
        num_bins = self.max_sequence_length // bin_size_bp
        if num_bins < 2:
            raise ValueError(
                f"Binned regression requires at least 2 bins, but max_sequence_length "
                f"({self.max_sequence_length}) / bin_size_bp ({bin_size_bp}) gives {num_bins}"
            )
        return num_bins

    def _predict_batch(
        self,
        model: Any,
        sequences: Any,
        conditional_input: Any,
        num_labels: int,
    ) -> Optional[np.ndarray]:
        """Return regression predictions for a batch, or None if the model does not provide them.

        The model is queried through ``_safe_model_call``, so a model that does not
        implement ``infer_sequence_to_regression_outputs`` (or returns None) is skipped
        and its metrics are None.

        Shape: [batch_size, num_labels] for sequence-level tasks,
        [batch_size, num_bins, num_labels] for binned tasks.
        """
        (preds,) = self._safe_model_call(
            model,
            "infer_sequence_to_regression_outputs",
            sequences,
            conditional_input,
            num_outputs=1,
        )
        if preds is None:
            return None
        preds = np.asarray(preds)
        if preds.shape[-1] != num_labels:
            raise ValueError(f"Expected {num_labels} labels, but got {preds.shape[-1]}")
        return preds

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on the given dataset.

        Args:
            model: Model instance to evaluate (must implement infer_sequence_to_regression_outputs;
                see the class docstring for the output contract)
            dataset: The dataset to evaluate on.

        Returns:
            dict: Scores with metric names as keys:
                - 'regression_pearsonr_macro': Pearson r, averaged over labels
                - 'regression_r2_macro': R^2 score, averaged over labels
        """
        # Create dataloader from dataset
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # Initialize metric classes (both handle sequence-level and binned predictions)
        metrics = [RegressionPearsonR(), RegressionR2()]
        num_bins = self._get_num_bins()
        num_labels = self._validate_num_labels()
        is_binned = num_bins > 1
        expected_ndim = 3 if is_binned else 2
        layout = "binned" if is_binned else "sequence"

        for sequences, labels, conditional_input in tqdm(data_loader, desc="Evaluating"):
            # Shape: [batch_size, num_labels] for sequence-level tasks,
            # [batch_size, num_bins, num_labels] for binned tasks
            preds = self._predict_batch(model, sequences, conditional_input, num_labels)
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            # Verify model output is valid
            if preds is not None:
                # Verify that the predictions match the expected layout, the target
                # shape, and the number of labels
                if preds.ndim != expected_ndim:
                    raise ValueError(
                        f"Expected {layout} regression predictions with "
                        f"{expected_ndim} dimensions, but got shape {preds.shape}"
                    )
                if preds.shape != labels_np.shape:
                    raise ValueError(
                        f"Prediction shape {preds.shape} does not match target shape "
                        f"{labels_np.shape}"
                    )
                if is_binned and preds.shape[1] != num_bins:
                    raise ValueError(
                        f"Expected {num_bins} bins, but got {preds.shape[1]}"
                    )

            # Calculate intermediate values for each metric
            for metric in metrics:
                metric.calc(preds, labels_np)

        # Get final results dynamically using metric.name as the result key
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def get_bin_size_bp(self) -> Optional[int]:
        """Return a positive bin width in bp, or None for sequence-level outputs.

        Binned tasks must use a fixed effective ``max_sequence_length`` that is
        an exact multiple of this width and yields at least two bins. Their
        datasets and predictions must have shape ``[batch, num_bins, num_labels]``.
        Sequence-level outputs have shape ``[batch, num_labels]`` and report
        ``num_bins = 1`` without a bin dimension.
        """
        pass
