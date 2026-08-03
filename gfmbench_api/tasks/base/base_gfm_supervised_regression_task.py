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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import RegressionPearsonR, RegressionR2
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class OutputSpatiality(str, Enum):
    SEQUENCE = "sequence"
    BINNED = "binned"


class BaseGFMSupervisedRegressionTask(BaseGFMTask):
    """Base class for sequence-level and spatially binned regression tasks.

    ``output_spatiality`` determines the target and prediction layout:

    * ``"sequence"``: ``[batch_size, num_outputs]``
    * ``"binned"``: ``[batch_size, num_bins, num_outputs]``

    Subclasses set ``output_spatiality`` and implement ``_get_num_outputs`` in
    addition to the abstract methods inherited from :class:`BaseGFMTask`.
    """

    output_spatiality: OutputSpatiality

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return regression objective and output-layout attributes."""
        if not isinstance(self.output_spatiality, OutputSpatiality):
            raise TypeError(
                "output_spatiality must be an OutputSpatiality member, "
                f"got {self.output_spatiality!r}"
            )
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "num_outputs": self._get_num_outputs(),
            "task_type": "regression",
            "output_spatiality": self.output_spatiality.value,
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Evaluate regression predictions with macro Pearson r and macro R^2."""
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        metrics = [RegressionPearsonR(), RegressionR2()]
        expected_ndim = (
            3 if self.output_spatiality == OutputSpatiality.BINNED else 2
        )

        for sequences, labels, conditional_input in tqdm(data_loader, desc="Evaluating"):
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
                preds = np.asarray(preds)
                assert preds.ndim == expected_ndim, (
                    f"Expected {self.output_spatiality.value} regression predictions with "
                    f"{expected_ndim} dimensions, but got shape {preds.shape}"
                )
                assert preds.shape == labels_np.shape, (
                    f"Prediction shape {preds.shape} does not match target shape "
                    f"{labels_np.shape}"
                )
                num_outputs = self._get_num_outputs()
                assert preds.shape[-1] == num_outputs, (
                    f"Expected {num_outputs} outputs, but got {preds.shape[-1]}"
                )

            for metric in metrics:
                metric.calc(preds, labels_np)

        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_num_outputs(self) -> int:
        """Return the number of regression outputs per sequence or bin."""
        pass
