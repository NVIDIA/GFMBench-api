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
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import (
    ClassificationAccuracy,
    ClassificationAUPRC,
    ClassificationAUROC,
    ClassificationMCC,
    MultiLabelClassificationAUPRC,
    MultiLabelClassificationAUROC,
)
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class ClassificationMode(str, Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


class InputStructure(str, Enum):
    SEQUENCE = "sequence"
    VARIANT_REFERENCE_PAIR = "variant_reference_pair"


class BaseGFMSupervisedClassificationTask(BaseGFMTask):
    """Base class for supervised single-label and multi-label classification.

    ``classification_mode`` controls target semantics and probability
    normalization. ``input_structure`` controls the dataset tuple and model
    inference method.
    """

    classification_mode: ClassificationMode
    input_structure: InputStructure

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def _get_num_labels(self) -> int:
        """Return independent target count; single-label tasks have one target."""
        if self.classification_mode == ClassificationMode.SINGLE_LABEL:
            return 1
        raise NotImplementedError("Multi-label tasks must implement _get_num_labels()")

    def _get_num_classes(self) -> int:
        """Return classes per target; conventional multi-label targets are binary."""
        if self.classification_mode == ClassificationMode.MULTI_LABEL:
            return 2
        raise NotImplementedError("Single-label tasks must implement _get_num_classes()")

    def _get_output_dim(self) -> int:
        """Return projection-head width for the configured classification mode."""
        if self.classification_mode == ClassificationMode.SINGLE_LABEL:
            return self._get_num_classes()
        return self._get_num_labels()

    def _validate_classification_attributes(self) -> None:
        if not isinstance(self.classification_mode, ClassificationMode):
            raise TypeError(
                "classification_mode must be a ClassificationMode member, "
                f"got {self.classification_mode!r}"
            )
        if not isinstance(self.input_structure, InputStructure):
            raise TypeError(
                "input_structure must be an InputStructure member, "
                f"got {self.input_structure!r}"
            )

        num_labels = self._get_num_labels()
        num_classes = self._get_num_classes()
        if num_labels < 1:
            raise ValueError(f"num_labels must be positive, got {num_labels}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if self.classification_mode == ClassificationMode.SINGLE_LABEL and num_labels != 1:
            raise ValueError("single_label classification must have num_labels == 1")
        if self.classification_mode == ClassificationMode.MULTI_LABEL and num_classes != 2:
            raise ValueError("multi_label classification requires binary labels")

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return classification semantics and input-layout attributes."""
        self._validate_classification_attributes()
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "task_type": "classification",
            "classification_mode": self.classification_mode.value,
            "input_structure": self.input_structure.value,
            "num_labels": self._get_num_labels(),
            "num_classes": self._get_num_classes(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Run the model method selected by input structure and label mode."""
        is_multilabel = self.classification_mode == ClassificationMode.MULTI_LABEL

        if self.input_structure == InputStructure.SEQUENCE:
            sequences, labels, conditional_input = batch
            method = (
                "infer_sequence_to_multilabel_probs"
                if is_multilabel
                else "infer_sequence_to_labels_probs"
            )
            probs, = self._safe_model_call(
                model, method, sequences, conditional_input, num_outputs=1
            )
            return probs, labels

        variant_sequences, ref_sequences, labels, conditional_input = batch
        method = (
            "infer_variant_ref_sequences_to_multilabel_probs"
            if is_multilabel
            else "infer_variant_ref_sequences_to_labels_probs"
        )
        probs, = self._safe_model_call(
            model,
            method,
            variant_sequences,
            ref_sequences,
            conditional_input,
            num_outputs=1,
        )
        return probs, labels

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Evaluate using metrics selected by classification mode."""
        self._validate_classification_attributes()
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        if self.classification_mode == ClassificationMode.MULTI_LABEL:
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
            probs, labels = self._batch_to_probs(batch, model)
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            if probs is not None:
                probs = np.asarray(probs)
                output_dim = self._get_output_dim()
                assert probs.ndim == 2 and probs.shape[1] == output_dim, (
                    f"Expected probabilities with shape [batch_size, {output_dim}], "
                    f"but got {probs.shape}"
                )
                if self.classification_mode == ClassificationMode.SINGLE_LABEL:
                    prob_sums = probs.sum(axis=1)
                    assert np.allclose(
                        prob_sums, np.ones_like(prob_sums), atol=1e-5
                    ), (
                        "Single-label probabilities do not sum to 1. "
                        f"Got range [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
                    )

            for metric in metrics:
                metric.calc(probs, labels_np)

        return {metric.name: metric.get_final_results() for metric in metrics}
