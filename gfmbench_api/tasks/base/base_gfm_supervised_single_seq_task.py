# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
from typing import Any, Optional, Tuple

import numpy as np

from gfmbench_api.tasks.base.base_gfm_supervised_multiclass_task import BaseGFMSupervisedMultiClassTask


class BaseGFMSupervisedSingleSeqTask(BaseGFMSupervisedMultiClassTask):
    """
    Base class for single-sequence classification tasks.
    
    Dataset format: (sequence, label, conditional_input) tuples
    Model inference: infer_sequence_to_labels_probs(sequences, conditional_input)
    
    Subclasses must implement:
        - _get_num_labels(): Return number of classification labels
        - _create_datasets(): Return train, validation (or None), test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def _is_variant_effect_prediction(self) -> bool:
        """Return False since this is a single-sequence task, not variant effect."""
        return False

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract probabilities and labels from a single-sequence batch (the common approach).
        
        Args:
            batch: Tuple of (sequences, labels, conditional_input) from DataLoader
            model: Model instance with infer_sequence_to_labels_probs method
        
        Returns:
            Tuple of (probs, labels_np):
                - probs: np.ndarray of shape [batch_size, num_labels] or None
                - labels_np: np.ndarray of shape [batch_size]
        """
        sequences, labels, conditional_input = batch
        
        # Get probabilities from model (returns numpy arrays)
        # Shape: [batch_size, num_labels] where num_labels = self._get_num_labels()
        probs, = self._safe_model_call(
            model, 'infer_sequence_to_labels_probs', sequences, conditional_input, num_outputs=1
        )
        
        return probs, labels

