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
    """
    Base class for supervised classification tasks.
    Implements testing for both single-label and multi-label classification tasks.

    The classification mode is derived from the number of labels: a task with one
    label is single-label (one distribution over num_classes classes), and a task
    with several labels is multi-label (one independent binary target per label).

    Subclasses must implement:
        - _get_input_structure(): Return the input structure of the task
        - _batch_to_probs(batch, model): Extract probs and labels from a batch
        - _get_num_labels(): Return number of independent classification targets
        - _get_num_classes(): Return number of classes per target
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
    """

    @abstractmethod
    def _get_num_classes(self) -> int:
        """Subclasses must implement this: return number of classes per target"""
        pass

    def _get_classification_mode(self) -> ClassificationMode:
        """Return single-label for tasks with one label, multi-label otherwise."""
        if self._get_num_labels() == 1:
            return ClassificationMode.SINGLE_LABEL
        return ClassificationMode.MULTI_LABEL

    def _get_output_dim(self) -> int:
        """
        Return the expected width of the model output.

        Single-label tasks output one probability per class, multi-label tasks
        output one probability per label.
        """
        if self._get_classification_mode() == ClassificationMode.SINGLE_LABEL:
            return self._get_num_classes()
        return self._get_num_labels()

    def _validate_classification_methods(self) -> None:
        """Verify that the values declared by the subclass are consistent."""
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
        """Return task attributes for classification tasks."""
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
        """
        Evaluate the model on the given dataset.

        Args:
            model: Model instance to evaluate (must implement the appropriate inference method)
            dataset: The dataset to evaluate on.

        Returns:
            dict: Scores with metric names as keys.
                For single-label tasks:
                - 'classification_accuracy': Accuracy score (0-1)
                - 'classification_mcc': Matthews Correlation Coefficient
                - 'classification_auroc': Area Under ROC Curve
                - 'classification_auprc': Area Under Precision-Recall Curve
                For multi-label tasks:
                - 'multilabel_auroc_macro': Area Under ROC Curve, averaged over labels
                - 'multilabel_auprc_macro': Area Under Precision-Recall Curve, averaged over labels
        """
        self._validate_classification_methods()
        classification_mode = self._get_classification_mode()

        # Create dataloader from dataset
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # Initialize metric classes matching the classification mode
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
            # Delegate to subclass for batch processing
            probs, labels = self._batch_to_probs(batch, model)
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            # Verify model output is valid
            if probs is not None:
                # Verify that the number of predictions matches the expected output width
                probs = np.asarray(probs)
                output_dim = self._get_output_dim()
                assert probs.ndim == 2 and probs.shape[1] == output_dim, (
                    f"Expected probabilities with shape [batch_size, {output_dim}], "
                    f"but got {probs.shape}"
                )
                if classification_mode == ClassificationMode.SINGLE_LABEL:
                    # Verify that probabilities sum to 1 (with epsilon tolerance).
                    # Multi-label probabilities are independent, so they are not checked.
                    prob_sums = probs.sum(axis=1)
                    assert np.allclose(
                        prob_sums, np.ones_like(prob_sums), atol=1e-5
                    ), (
                        "Single-label probabilities do not sum to 1. "
                        f"Got range [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
                    )

            # Calculate intermediate values for each metric
            for metric in metrics:
                metric.calc(probs, labels_np)

        # Get final results dynamically using metric.name as the result key
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_input_structure(self) -> InputStructure:
        """Subclasses must implement this: return the input structure of the task"""
        pass

    @abstractmethod
    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract probabilities and labels from a batch using the model.

        Subclasses must implement this to handle their specific batch format:
        - Single sequence tasks: batch = (sequences, labels, conditional_input)
        - Variant effect tasks: batch = (variant_sequences, ref_sequences, labels, conditional_input)

        Args:
            batch: A batch from the DataLoader (tuple of tensors/lists)
            model: Model instance to use for inference

        Returns:
            Tuple of (probs, labels):
                - probs: np.ndarray of shape [batch_size, num_classes] for single-label
                  tasks, [batch_size, num_labels] for multi-label tasks, or None
                - labels: labels of shape [batch_size] for single-label tasks,
                  [batch_size, num_labels] for multi-label tasks
        """
        pass
