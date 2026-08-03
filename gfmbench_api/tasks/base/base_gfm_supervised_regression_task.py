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
    """Base class for sequence-level and spatially binned regression tasks.

    Each example is ``(sequence, labels, conditional_input)``. The value
    returned by ``_get_output_spatiality`` determines the label and prediction
    layout:

    * ``SEQUENCE``: ``[batch_size, num_labels]``
    * ``BINNED``: ``[batch_size, num_bins, num_labels]``

    Evaluation calls ``infer_sequence_to_regression`` and reports macro Pearson
    r and macro R². For binned outputs, leading sample and bin dimensions are
    pooled per label by the metrics.

    Subclasses implement ``_get_output_spatiality`` and the inherited
    ``_get_num_labels`` method, together with dataset creation, task naming,
    default sequence length and conditional-input metadata methods.
    """

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return regression objective and output-layout attributes."""
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
        """Evaluate regression predictions with macro Pearson r and macro R².

        Args:
            model: Model implementing ``infer_sequence_to_regression``.
            dataset: Dataset yielding sequence, label and conditional-input
                tuples.

        Returns:
            Metric names mapped to their final scalar scores.
        """
        # Create a DataLoader from the evaluation dataset.
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # Both metrics support sequence-level and binned regression arrays.
        metrics = [RegressionPearsonR(), RegressionR2()]
        output_spatiality = self._get_output_spatiality()
        expected_ndim = 3 if output_spatiality == OutputSpatiality.BINNED else 2

        for sequences, labels, conditional_input in tqdm(data_loader, desc="Evaluating"):
            # Shape is [batch, labels] or [batch, bins, labels], according to
            # the spatiality declared by the concrete task.
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

            if preds is not None:
                # Verify dimensionality, target agreement and label count.
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

            # Accumulate intermediate values for every regression metric.
            for metric in metrics:
                metric.calc(preds, labels_np)

        # Use each metric's public name as its result key.
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_output_spatiality(self) -> OutputSpatiality:
        """Return whether labels are sequence-level or spatially binned."""
        pass
