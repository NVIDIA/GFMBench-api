# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import roc_auc_score
from .base_metric import BaseMetric


class SequenceEmbeddingsL2AUROC(BaseMetric):
    """
    Sequence embeddings L2 distance based AUROC metric.
    
    Receives representative embeddings for variant and reference sequences (e.g., CLS token 
    for BERT, last token for GPT) and ground truth labels. Computes AUROC based on 
    L2 (Euclidean) distance between sequence representatives.
    
    Higher L2 distance indicates pathogenic variants (more different from reference).
    """
    
    def __init__(self):
        """Initialize storage for variant/reference representations and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._l2_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "sequence_embeddings_l2_auroc"
    
    def _calc_impl(self, variant_repr, reference_repr, gt):
        """Calculate and store L2 distance scores."""
        # Compute L2 (Euclidean) distance between variant and reference embeddings
        # [batch_size, hidden_dim] -> [batch_size]
        l2_distance = np.linalg.norm(variant_repr - reference_repr, axis=1)
        
        # Higher L2 distance (more different) indicates pathogenic variants (label=1)
        # So we use L2 distance directly as the prediction score
        self._l2_scores_list.append(l2_distance)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUROC from stored L2 distance scores."""
        if not self._l2_scores_list:
            return None
        
        # Concatenate all batches
        all_l2_scores = np.concatenate(self._l2_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUROC
        return roc_auc_score(all_gt, all_l2_scores)

