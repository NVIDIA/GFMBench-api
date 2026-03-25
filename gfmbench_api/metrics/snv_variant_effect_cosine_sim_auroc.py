# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import roc_auc_score
from .base_metric import BaseMetric


class SNVVariantEffectCosineSimAUROC(BaseMetric):
    """
    SNV (Single Nucleotide Variant) position variant effect AUROC metric.
    
    Receives embeddings of all variant and reference samples, per-sequence indices of the SNV position,
    and ground truth labels. Performs AUROC on the 1-cosine similarity (or negative cosine similarity).
    
    This metric is used for variant effect prediction tasks where we want to measure
    how well the model distinguishes between pathogenic and benign variants based on
    embedding similarity at the variant position.
    
    For tokenizers like BPE where the output position varies per sequence, this metric
    handles per-sequence position arrays.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference embeddings, SNV indices, and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._cosine_sim_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "snv_variant_effect_cosinesim_auroc"
    
    def _calc_impl(self, variant_embeddings, reference_embeddings, variant_snv_indices, reference_snv_indices, gt):
        """Calculate and store negative cosine similarity at SNV position for each sequence.
        
        Args:
            variant_embeddings: [batch_size, seq_len, hidden_dim] embeddings for variant sequences
            reference_embeddings: [batch_size, seq_len, hidden_dim] embeddings for reference sequences
            variant_snv_indices: [batch_size] array of SNV positions in variant sequences
            reference_snv_indices: [batch_size] array of SNV positions in reference sequences
            gt: [batch_size] ground truth labels
        """
        batch_size = variant_embeddings.shape[0]
        hidden_dim = variant_embeddings.shape[2]
        
        # Initialize arrays for extracted embeddings
        variant_snv_emb = np.zeros((batch_size, hidden_dim))
        reference_snv_emb = np.zeros((batch_size, hidden_dim))
        valid_mask = np.ones(batch_size, dtype=bool)
        
        # Extract embeddings at the specific SNV position for each sequence
        for i in range(batch_size):
            var_idx = variant_snv_indices[i]
            ref_idx = reference_snv_indices[i]
            
            # Check if indices are valid and within bounds
            if (var_idx is None or var_idx < 0 or var_idx >= variant_embeddings.shape[1] or
                ref_idx is None or ref_idx < 0 or ref_idx >= reference_embeddings.shape[1]):
                valid_mask[i] = False
                continue
            
            # Extract embeddings at SNV position
            variant_snv_emb[i] = variant_embeddings[i, var_idx, :]
            reference_snv_emb[i] = reference_embeddings[i, ref_idx, :]
        
        # If no valid samples, return
        if not np.any(valid_mask):
            return
        
        # Filter to only valid samples
        variant_snv_emb = variant_snv_emb[valid_mask]
        reference_snv_emb = reference_snv_emb[valid_mask]
        gt = gt[valid_mask]
        
        # Normalize embeddings to unit vectors for cosine similarity
        variant_snv_norm = variant_snv_emb / (np.linalg.norm(variant_snv_emb, axis=1, keepdims=True) + 1e-10)
        reference_snv_norm = reference_snv_emb / (np.linalg.norm(reference_snv_emb, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity at SNV position
        # Cosine similarity = dot product of normalized vectors
        cosine_sim = np.sum(variant_snv_norm * reference_snv_norm, axis=1)  # [batch_size]
        
        # Lower cosine similarity (less similar) indicates pathogenic variants (label=1)
        # So we use negative cosine similarity as the prediction score
        # This way, higher scores indicate higher probability of being pathogenic
        neg_cosine_sim = -cosine_sim
        
        # Store only the cosine similarity scores and labels
        self._cosine_sim_scores_list.append(neg_cosine_sim)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUROC from stored cosine similarity scores at SNV position."""
        if not self._cosine_sim_scores_list:
            return None
        
        # Concatenate all batches
        all_cosine_sim_scores = np.concatenate(self._cosine_sim_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUROC
        return roc_auc_score(all_gt, all_cosine_sim_scores)

