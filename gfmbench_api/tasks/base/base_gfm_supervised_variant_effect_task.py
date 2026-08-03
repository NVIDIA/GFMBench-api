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
    """Base class for classification from variant/reference sequence pairs.

    Dataset format:
        ``(variant_sequence, reference_sequence, label, conditional_input)``
        tuples. The variant-first order is consistent with zero-shot variant
        tasks.

    Model inference:
        - Single-label: ``infer_variant_ref_sequences_to_labels_probs``
        - Multi-label: ``infer_variant_ref_sequences_to_multilabel_probs``

    Subclasses must implement ``_get_num_labels``, ``_get_num_classes``,
    ``_create_datasets``, ``get_task_name``, ``_get_default_max_seq_len`` and
    ``get_conditional_input_meta_data_frame``.
    """

    def _get_input_structure(self) -> InputStructure:
        """Return the variant/reference-pair input layout."""
        return InputStructure.VARIANT_REFERENCE_PAIR

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Extract probabilities and labels from a variant/reference batch.

        Args:
            batch: ``(variant_sequences, reference_sequences, labels,
                conditional_input)`` from a DataLoader.
            model: Model implementing the paired-sequence inference method
                appropriate for the derived classification mode.

        Returns:
            A ``(probs, labels)`` tuple. ``probs`` is either ``None`` or an
            array shaped ``[batch_size, output_dim]``; ``labels`` is returned
            unchanged for conversion by the shared evaluation loop.
        """
        variant_sequences, ref_sequences, labels, conditional_input = batch

        # Multi-label tasks produce one independent probability per label;
        # single-label tasks produce a normalized distribution over classes.
        method = (
            "infer_variant_ref_sequences_to_multilabel_probs"
            if self._get_classification_mode() == ClassificationMode.MULTI_LABEL
            else "infer_variant_ref_sequences_to_labels_probs"
        )

        # Model inference methods return probabilities as NumPy arrays.
        probs, = self._safe_model_call(
            model,
            method,
            variant_sequences,
            ref_sequences,
            conditional_input,
            num_outputs=1,
        )
        return probs, labels
