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

from typing import Any, Optional, Tuple

import numpy as np

from gfmbench_api.tasks.base.base_gfm_supervised_classification_task import (
    BaseGFMSupervisedClassificationTask,
    ClassificationMode,
    InputStructure,
)


class BaseGFMSupervisedVariantEffectTask(BaseGFMSupervisedClassificationTask):
    """
    Base class for variant effect classification tasks with variant/reference sequence pairs.

    Dataset format: (variant_sequence, ref_sequence, label, conditional_input) tuples
    Model inference: infer_variant_ref_sequences_to_labels_probs(variant_sequences,
        ref_sequences, conditional_input) for single-label tasks,
        infer_variant_ref_sequences_to_multilabel_probs(variant_sequences, ref_sequences,
        conditional_input) for multi-label tasks

    Note: Order is variant first, then reference - consistent with zero-shot variant tasks.

    Subclasses must implement:
        - _get_num_labels(): Return number of independent classification targets
        - _get_num_classes(): Return number of classes per target
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def _get_input_structure(self) -> InputStructure:
        """Return the input structure of this task: a variant/reference pair per example."""
        return InputStructure.VARIANT_REFERENCE_PAIR

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract probabilities and labels from a variant-reference sequence batch.

        Args:
            batch: Tuple of (variant_sequences, ref_sequences, labels, conditional_input) from DataLoader
            model: Model instance with infer_variant_ref_sequences_to_labels_probs method
                (single-label) or infer_variant_ref_sequences_to_multilabel_probs method (multi-label)

        Returns:
            Tuple of (probs, labels):
                - probs: np.ndarray of shape [batch_size, num_classes] for single-label
                  tasks, [batch_size, num_labels] for multi-label tasks, or None
                - labels: labels of shape [batch_size] for single-label tasks,
                  [batch_size, num_labels] for multi-label tasks
        """
        variant_sequences, ref_sequences, labels, conditional_input = batch

        # Single-label tasks predict one distribution over classes, multi-label tasks
        # predict one independent probability per label.
        method = (
            "infer_variant_ref_sequences_to_multilabel_probs"
            if self._get_classification_mode() == ClassificationMode.MULTI_LABEL
            else "infer_variant_ref_sequences_to_labels_probs"
        )

        # Get probabilities from model (returns numpy arrays)
        # Shape: [batch_size, output_dim] where output_dim = self._get_output_dim()
        probs, = self._safe_model_call(
            model,
            method,
            variant_sequences,
            ref_sequences,
            conditional_input,
            num_outputs=1,
        )

        return probs, labels
