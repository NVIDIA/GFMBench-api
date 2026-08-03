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
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmbench_api.metrics import RegressionPearsonR, RegressionR2
from gfmbench_api.tasks.base.base_gfm_supervised_task import BaseGFMSupervisedTask


class OutputSpatiality(str, Enum):
    SEQUENCE = "sequence"
    BINNED = "binned"


class BaseGFMSupervisedRegressionTask(BaseGFMSupervisedTask):
    """
    Base class for sequence-level and binned regression tasks (e.g. CAGE).

    Dataset format: (sequence, label, conditional_input) tuples
    Model inference: infer_sequence_to_regression(sequences, conditional_input)

    The output spatiality determines the label and prediction layout:
        - 'sequence': [batch_size, num_labels] - one value per label per sequence
        - 'binned': [batch_size, num_bins, num_labels] - one value per label per bin,
          where num_bins = sequence_length / bin_size_bp

    Evaluation reports macro Pearson r and macro R^2, computed per label after pooling
    all leading dimensions (samples, and bins for binned tasks).

    Subclasses must implement:
        - _get_output_spatiality(): Return 'sequence' or 'binned' output layout
        - _get_num_labels(): Return number of regression outputs per sequence or bin
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for regression tasks."""
        output_spatiality = self._get_output_spatiality()
        if not isinstance(output_spatiality, OutputSpatiality):
            raise TypeError(
                "_get_output_spatiality() must return an OutputSpatiality member, "
                f"got {output_spatiality!r}"
            )
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "num_labels": self._validate_num_labels(),
            "task_type": "regression",
            "output_spatiality": output_spatiality.value,
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on the given dataset.

        Args:
            model: Model instance to evaluate (must implement infer_sequence_to_regression)
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
        output_spatiality = self._get_output_spatiality()
        expected_ndim = 3 if output_spatiality == OutputSpatiality.BINNED else 2

        for sequences, labels, conditional_input in tqdm(data_loader, desc="Evaluating"):
            # Shape: [batch_size, num_labels] for sequence-level tasks,
            # [batch_size, num_bins, num_labels] for binned tasks
            preds, = self._safe_model_call(
                model,
                "infer_sequence_to_regression",
                sequences,
                conditional_input,
                num_outputs=1,
            )
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            # Verify model output is valid
            if preds is not None:
                # Verify that the predictions match the expected layout, the target
                # shape, and the number of labels
                preds = np.asarray(preds)
                assert preds.ndim == expected_ndim, (
                    f"Expected {output_spatiality.value} regression predictions with "
                    f"{expected_ndim} dimensions, but got shape {preds.shape}"
                )
                assert preds.shape == labels_np.shape, (
                    f"Prediction shape {preds.shape} does not match target shape "
                    f"{labels_np.shape}"
                )
                num_labels = self._get_num_labels()
                assert preds.shape[-1] == num_labels, (
                    f"Expected {num_labels} labels, but got {preds.shape[-1]}"
                )

            # Calculate intermediate values for each metric
            for metric in metrics:
                metric.calc(preds, labels_np)

        # Get final results dynamically using metric.name as the result key
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_output_spatiality(self) -> OutputSpatiality:
        """Subclasses must implement this: return the output layout of the task"""
        pass
