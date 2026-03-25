# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import accuracy_score
from .base_metric import BaseMetric


class MultiLabelClassificationAccuracy(BaseMetric):
    """
    Multi-label classification accuracy metric.
    
    Receives probabilities per label for the entire dataset and ground truth labels.
    Performs argmax on probabilities and compares to ground truth for accuracy score.
    """
    
    def __init__(self):
        """Initialize storage for probabilities and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._predictions_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "classification_accuracy"
    
    def _calc_impl(self, probs, gt):
        """Compute predictions via argmax and store them."""
        # Perform argmax to get predicted labels
        predictions = np.argmax(probs, axis=1)
        
        # Store only predictions and labels
        self._predictions_list.append(predictions)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate classification accuracy from stored predictions."""
        if not self._predictions_list:
            return None
        
        # Concatenate all batches
        all_predictions = np.concatenate(self._predictions_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate accuracy
        return accuracy_score(all_gt, all_predictions)

