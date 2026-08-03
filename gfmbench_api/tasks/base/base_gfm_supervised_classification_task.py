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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmbench_api.metrics import (
    ClassificationAccuracy,
    ClassificationAUPRC,
    ClassificationAUROC,
    ClassificationMCC,
    MultiLabelClassificationAUPRC,
    MultiLabelClassificationAUROC,
)
from gfmbench_api.tasks.base.base_gfm_supervised_task import BaseGFMSupervisedTask


class ClassificationMode(str, Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


class InputStructure(str, Enum):
    SEQUENCE = "sequence"
    VARIANT_REFERENCE_PAIR = "variant_reference_pair"


class BaseGFMSupervisedClassificationTask(BaseGFMSupervisedTask):
    """Base class for supervised single-label and multi-label classification.

    Classification mode is derived from ``_get_num_labels``: one label is
    single-label classification, while multiple labels are independent binary
    targets. Input routing is supplied by a concrete classification base.

    Subclasses must implement ``_get_num_labels``, ``_get_num_classes``,
    ``_create_datasets``, ``get_task_name``, ``_get_default_max_seq_len`` and
    ``get_conditional_input_meta_data_frame``. The single-sequence and
    variant-effect bases provide the input-specific batch routing.
    """

    @abstractmethod
    def _get_num_classes(self) -> int:
        """Return classes per label (two for independent multi-label targets)."""
        pass

    def _get_classification_mode(self) -> ClassificationMode:
        """Derive single-label or multi-label semantics from label count."""
        if self._get_num_labels() == 1:
            return ClassificationMode.SINGLE_LABEL
        return ClassificationMode.MULTI_LABEL

    def _get_output_dim(self) -> int:
        """Return projection-head width for the derived classification mode."""
        if self._get_classification_mode() == ClassificationMode.SINGLE_LABEL:
            return self._get_num_classes()
        return self._get_num_labels()

    def _validate_classification_methods(self) -> None:
        """Validate values returned by the classification contract methods."""
        input_structure = self._get_input_structure()
        if not isinstance(input_structure, InputStructure):
            raise TypeError(
                "_get_input_structure() must return an InputStructure member, "
                f"got {input_structure!r}"
            )

        num_labels = self._validate_num_labels()
        num_classes = self._get_num_classes()
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if num_labels > 1 and num_classes != 2:
            raise ValueError("multi_label classification requires binary labels")

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return classification semantics and input-layout attributes."""
        self._validate_classification_methods()
        classification_mode = self._get_classification_mode()
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "task_type": "classification",
            "classification_mode": classification_mode.value,
            "input_structure": self._get_input_structure().value,
            "num_labels": self._get_num_labels(),
            "num_classes": self._get_num_classes(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Evaluate a classification dataset.

        Args:
            model: Model implementing the inference method selected by the
                concrete input-routing base.
            dataset: Dataset to evaluate.

        Returns:
            Single-label tasks report accuracy, MCC, AUROC and AUPRC.
            Multi-label tasks report macro AUROC and macro AUPRC.
        """
        self._validate_classification_methods()
        classification_mode = self._get_classification_mode()

        # Create a DataLoader from the evaluation dataset.
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # Select metrics that match the derived target semantics.
        if classification_mode == ClassificationMode.MULTI_LABEL:
            metrics = [
                MultiLabelClassificationAUROC(),
                MultiLabelClassificationAUPRC(),
            ]
        else:
            metrics = [
                ClassificationAccuracy(),
                ClassificationMCC(),
                ClassificationAUROC(),
                ClassificationAUPRC(),
            ]

        for batch in tqdm(data_loader, desc="Evaluating"):
            # Delegate input-specific unpacking and inference to the child base.
            probs, labels = self._batch_to_probs(batch, model)
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            if probs is not None:
                # Verify that the model output matches the expected head width.
                probs = np.asarray(probs)
                output_dim = self._get_output_dim()
                assert probs.ndim == 2 and probs.shape[1] == output_dim, (
                    f"Expected probabilities with shape [batch_size, {output_dim}], "
                    f"but got {probs.shape}"
                )
                if classification_mode == ClassificationMode.SINGLE_LABEL:
                    # Single-label class probabilities must form a distribution.
                    prob_sums = probs.sum(axis=1)
                    assert np.allclose(
                        prob_sums, np.ones_like(prob_sums), atol=1e-5
                    ), (
                        "Single-label probabilities do not sum to 1. "
                        f"Got range [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
                    )

            # Accumulate intermediate values for every selected metric.
            for metric in metrics:
                metric.calc(probs, labels_np)

        # Use each metric's public name as its result key.
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_input_structure(self) -> InputStructure:
        """Return the classification input layout."""
        pass

    @abstractmethod
    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Extract probabilities and labels using input-specific inference.

        Concrete routing bases implement this for either
        ``(sequence, label, conditional_input)`` or
        ``(variant_sequence, reference_sequence, label, conditional_input)``
        batches.
        """
        pass
