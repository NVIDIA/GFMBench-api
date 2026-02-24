from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from gfmbench_api.metrics import (
    SNVVariantEffectCosineSimAUPRC,
    SNVVariantEffectCosineSimAUROC,
    SNVVariantEffectPredictionMaskedLLRAUPRC,
    SNVVariantEffectPredictionMaskedLLRAUROC,
)
from gfmbench_api.tasks.base.base_gfm_zero_shot_task import BaseGFMZeroShotTask


class BaseGFMZeroShotSNVTask(BaseGFMZeroShotTask):
    """
    Base class for zero-shot tasks specifically for Single Nucleotide Variants (SNV).
    Extends BaseGFMZeroShotTask with SNV-specific metrics.
    
    This class should be used for datasets containing only SNV variants, where
    position-specific variant effect metrics can be calculated.
    
    Additional metrics (beyond common ones):
        - SNV position cosine similarity (BEND-style)
        - Masked token prediction LLR
    
    Subclasses must implement:
        - _get_variant_position_in_sequence(): Return position of SNV in sequence
        - _create_test_dataset(): Return test dataset
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
    """

    def _get_additional_metrics(self) -> List[tuple]:
        """
        Return SNV-specific metrics with their argument keys.
        
        Each entry is a tuple: (metric, arg_keys)
        arg_keys specifies which outputs to pass to metric.calc()
        """
        return [
            (SNVVariantEffectCosineSimAUROC(), 
             ('variant_embeddings', 'reference_embeddings', 'variant_snv_pos', 'reference_snv_pos', 'labels')),
            (SNVVariantEffectCosineSimAUPRC(), 
             ('variant_embeddings', 'reference_embeddings', 'variant_snv_pos', 'reference_snv_pos', 'labels')),
            (SNVVariantEffectPredictionMaskedLLRAUROC(), 
             ('variant_token_probs', 'reference_token_probs', 'labels')),
            (SNVVariantEffectPredictionMaskedLLRAUPRC(), 
             ('variant_token_probs', 'reference_token_probs', 'labels')),
        ]

    def _update_additional_metrics(
        self,
        additional_metrics: List[tuple],
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        labels_np: np.ndarray,
        outputs: Dict[str, Optional[np.ndarray]]
    ) -> None:
        """
        Update SNV-specific metrics for a batch.
        
        Gets variant position, maps to output positions, and calculates
        masked token probabilities, then updates all metrics using arg_keys.
        """
        # Get the SNV position in the input sequence
        snv_input_pos = self._get_variant_position_in_sequence()
        
        # Map input position to output positions for each sequence in the batch
        variant_snv_output_pos, = self._safe_model_call(
            model, 'sequence_pos_to_prob_pos', variant_sequences, snv_input_pos, num_outputs=1
        )
        reference_snv_output_pos, = self._safe_model_call(
            model, 'sequence_pos_to_prob_pos', reference_sequences, snv_input_pos, num_outputs=1
        )

        # Get masked token probabilities for variant and reference nucleotides
        variant_letters = [seq[snv_input_pos] for seq in variant_sequences]
        reference_letters = [seq[snv_input_pos] for seq in reference_sequences]
        variant_token_probs_np, reference_token_probs_np = self._safe_model_call(
            model, 'infer_masked_sequence_to_token_probs', 
            variant_sequences, snv_input_pos, variant_letters, reference_letters,
            num_outputs=2
        )
        
        # Add SNV-specific outputs to the outputs dict
        outputs['variant_snv_pos'] = variant_snv_output_pos
        outputs['reference_snv_pos'] = reference_snv_output_pos
        outputs['variant_token_probs'] = variant_token_probs_np
        outputs['reference_token_probs'] = reference_token_probs_np
        
        # Update all SNV metrics using their specified argument keys
        for metric, arg_keys in additional_metrics:
            args = [outputs[key] for key in arg_keys]
            metric.calc(*args)

    def _is_snv_only(self) -> bool:
        """Return True since this handles only SNV variants."""
        return True

    @abstractmethod
    def _get_variant_position_in_sequence(self) -> int:
        """
        Subclasses must implement this: return the position of the SNV in the sequence.
        
        Returns:
            int: position of the variant in the input sequence (0-based)
        """
        pass
