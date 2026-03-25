# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
import numpy as np
from sklearn.metrics import average_precision_score
from .base_metric import BaseMetric


class SumProbsLLRAUPRC(BaseMetric):
    """
    Log-Likelihood Ratio (LLR) based AUPRC metric for variant effect prediction.
    
    Receives probabilities for variant and reference sequences and ground truth labels.
    Calculates LLR as log(P(variant)) - log(P(reference)) and computes AUPRC.
    
    Lower LLR indicates pathogenic variants, so we use -LLR for the metric calculation.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference probabilities and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._llr_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "sum_probs_llr_auprc"
    
    def _calc_impl(self, variant_probs, reference_probs, gt, epsilon=1e-10):
        """Calculate and store negative LLR scores."""
        # Verify probabilities are valid (between 0 and 1)
        assert np.all((variant_probs >= 0) & (variant_probs <= 1)), \
            "Variant probabilities must be between 0 and 1"
        assert np.all((reference_probs >= 0) & (reference_probs <= 1)), \
            "Reference probabilities must be between 0 and 1"
        
        # Calculate log-likelihood for all sequences in batch
        # Sum log probabilities across sequence dimension: [batch_size, seq_len] -> [batch_size]
        log_likelihood_variant = np.sum(np.log(variant_probs + epsilon), axis=1)
        log_likelihood_reference = np.sum(np.log(reference_probs + epsilon), axis=1)
        
        # Calculate LLR: [batch_size]
        llr_scores = log_likelihood_variant - log_likelihood_reference
        
        # We use -LLR because lower LLR indicates pathogenic variants (label=1)
        # and sklearn metrics expect higher scores to indicate the positive class
        neg_llr_scores = -llr_scores
        
        # Store only the LLR scores and labels
        self._llr_scores_list.append(neg_llr_scores)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUPRC from stored LLR scores."""
        if not self._llr_scores_list:
            return None
        
        # Concatenate all batches
        all_llr_scores = np.concatenate(self._llr_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUPRC
        return average_precision_score(all_gt, all_llr_scores)

