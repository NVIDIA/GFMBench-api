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
from typing import Any, Optional, Tuple

import numpy as np

from gfmbench_api.tasks.base.base_gfm_supervised_multiclass_task import BaseGFMSupervisedMultiClassTask


class BaseGFMSupervisedVariantEffectTask(BaseGFMSupervisedMultiClassTask):
    """
    Base class for variant effect classification tasks with variant/reference sequence pairs.
    
    Dataset format: (variant_sequence, ref_sequence, label, conditional_input) tuples
    Model inference: infer_variant_ref_sequences_to_labels_probs(variant_sequences, ref_sequences, conditional_input)
    
    Note: Order is variant first, then reference - consistent with zero-shot variant tasks.
    
    Subclasses must implement:
        - _get_num_labels(): Return number of classification labels
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def _is_variant_effect_prediction(self) -> bool:
        """Return True since this is a variant effect prediction task."""
        return True

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract probabilities and labels from a variant-reference sequence batch.
        
        Args:
            batch: Tuple of (variant_sequences, ref_sequences, labels, conditional_input) from DataLoader
            model: Model instance with infer_variant_ref_sequences_to_labels_probs method
        
        Returns:
            Tuple of (probs, labels_np):
                - probs: np.ndarray of shape [batch_size, num_labels] or None
                - labels_np: np.ndarray of shape [batch_size]
        """
        variant_sequences, ref_sequences, labels, conditional_input = batch
        
        # Get probabilities from model (returns numpy arrays)
        # Shape: [batch_size, num_labels] where num_labels = self._get_num_labels()
        probs, = self._safe_model_call(
            model, 'infer_variant_ref_sequences_to_labels_probs', 
            variant_sequences, ref_sequences, conditional_input, num_outputs=1
        )
        
        return probs, labels

