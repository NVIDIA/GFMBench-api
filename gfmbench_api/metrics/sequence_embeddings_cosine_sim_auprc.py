# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import average_precision_score
from .base_metric import BaseMetric


class SequenceEmbeddingsCosineSimAUPRC(BaseMetric):
    """
    Sequence embeddings cosine similarity based AUPRC metric.
    
    Receives representative embeddings for variant and reference sequences (e.g., CLS token 
    for BERT, last token for GPT) and ground truth labels. Computes AUPRC based on 
    cosine similarity between sequence representatives.
    
    Lower similarity indicates pathogenic variants, so we use negative cosine similarity.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference representations and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._cosine_sim_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "sequence_embeddings_cosinesim_auprc"
    
    def _calc_impl(self, variant_repr, reference_repr, gt):
        """Calculate and store negative cosine similarity scores."""
        # Normalize embeddings to unit vectors for cosine similarity
        # [batch_size, hidden_dim] -> [batch_size, hidden_dim]
        variant_repr_norm = variant_repr / (np.linalg.norm(variant_repr, axis=1, keepdims=True) + 1e-10)
        reference_repr_norm = reference_repr / (np.linalg.norm(reference_repr, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity: dot product of normalized vectors
        # [batch_size, hidden_dim] * [batch_size, hidden_dim] -> [batch_size]
        cosine_sim = np.sum(variant_repr_norm * reference_repr_norm, axis=1)
        
        # Lower cosine similarity (less similar) indicates pathogenic variants (label=1)
        # So we use negative cosine similarity as the prediction score
        neg_cosine_sim = -cosine_sim
        
        # Store only the cosine similarity scores and labels
        self._cosine_sim_scores_list.append(neg_cosine_sim)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUPRC from stored cosine similarity scores."""
        if not self._cosine_sim_scores_list:
            return None
        
        # Concatenate all batches
        all_cosine_sim_scores = np.concatenate(self._cosine_sim_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUPRC
        return average_precision_score(all_gt, all_cosine_sim_scores)

