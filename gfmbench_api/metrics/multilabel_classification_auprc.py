# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import average_precision_score
from .base_metric import BaseMetric


class MultiLabelClassificationAUPRC(BaseMetric):
    """
    Multi-label classification Area Under the Precision-Recall Curve (AUPRC) metric.
    
    Receives probabilities per label for the entire dataset and ground truth labels.
    Calculates AUPRC for binary or multi-class classification.
    """
    
    def __init__(self):
        """Initialize storage for probabilities and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._probs_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "classification_auprc"
    
    def _calc_impl(self, probs, gt):
        """Store probabilities and labels for AUPRC calculation."""
        # Store probabilities and labels as numpy arrays
        self._probs_list.append(probs)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUPRC from stored probabilities."""
        if not self._probs_list:
            return None
        
        # Concatenate all batches
        all_probs = np.concatenate(self._probs_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Determine number of classes
        num_classes = all_probs.shape[1]
        
        if num_classes == 2:
            # Binary classification: use probabilities of positive class
            return average_precision_score(all_gt, all_probs[:, 1])
        else:
            # Multi-class: use macro averaging
            return average_precision_score(all_gt, all_probs, average='macro')

