from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel


class GFMWithProjection(BaseGFMModel):
    """
    Wrapper class that combines a BaseGFMModel with a projection layer.
    The benchmark treats this as a single unified model.
    """
    
    def __init__(self, base_model: BaseGFMModel, projection_layer: Optional[nn.Module] = None) -> None:
        """
        Initialize the wrapped model.
        
        Args:
            base_model: BaseGFMModel instance
            projection_layer: nn.Module - projection layer (optional, for classification tasks)
        """
        self.base_model: BaseGFMModel = base_model
        self.projection_layer: Optional[nn.Module] = projection_layer
        self.device: str = base_model.device
    
    def infer_sequence_to_labels_probs(self, sequences: List[str], conditional_input=None) -> Optional[np.ndarray]:
        """
        Forward pass through the model and projection layer to get label probabilities.
        
        Args:
            sequences: list of DNA strings
            conditional_input: Optional metadata inputs (not used in this wrapper)
            
        Returns:
            np.ndarray: Probabilities of shape [batch_size, num_labels] if projection layer exists,
                       otherwise returns sequence representative embeddings of shape [batch_size, hidden_dim]
        """
        # Get embeddings from base model using infer_sequence_to_sequence
        _, _, sequence_repr_np = self.base_model.infer_sequence_to_sequence(sequences, conditional_input)
        
        if sequence_repr_np is None:
            return None
        
        # Apply projection layer if it exists
        if self.projection_layer is not None:
            with torch.no_grad():
                sequence_repr = torch.from_numpy(sequence_repr_np).to(self.device)
                logits = self.projection_layer(sequence_repr)
                # Apply softmax to convert logits to probabilities
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
        else:
            return sequence_repr_np
    
    def infer_sequence_to_sequence(self, sequences: List[str], conditional_input=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Delegate to base model for sequence-to-sequence inference.
        
        Args:
            sequences: list of DNA strings
            conditional_input: Optional metadata inputs
            
        Returns:
            tuple: (sequence_probs, sequence_embeddings, sequence_representative)
                - sequence_probs: np.ndarray [batch_size, seq_len] OR None
                - sequence_embeddings: np.ndarray [batch_size, seq_len, hidden_dim] OR None
                - sequence_representative: np.ndarray [batch_size, hidden_dim] OR None
        """
        return self.base_model.infer_sequence_to_sequence(sequences, conditional_input)
    
    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        """
        Delegate to the base model's sequence_pos_to_prob_pos method.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            pos: int - position in the input DNA sequence (0-based)
        
        Returns:
            np.ndarray: array of shape [batch_size] with output positions
        """
        return self.base_model.sequence_pos_to_prob_pos(sequences, pos)
    
    def eval(self) -> 'GFMWithProjection':
        """Set model to evaluation mode."""
        self.base_model.eval()
        if self.projection_layer is not None:
            self.projection_layer.eval()
        return self
    
    def train(self, mode: bool = True) -> 'GFMWithProjection':
        """Set model to training mode."""
        self.base_model.train(mode)
        if self.projection_layer is not None:
            self.projection_layer.train(mode)
        return self
    
    def to(self, device: str) -> 'GFMWithProjection':
        """Move model to device."""
        self.base_model.to(device)
        if self.projection_layer is not None:
            self.projection_layer.to(device)
        self.device = device
        return self
    
    def get_hidden_dim(self):
        """Return the hidden dimension of the base model."""
        return self.base_model.get_hidden_dim()
    
    def parameters(self) -> Iterator[nn.Parameter]:
        """Get all parameters from base model and projection layer."""
        params = list(self.base_model.parameters())
        if self.projection_layer is not None:
            params.extend(list(self.projection_layer.parameters()))
        return iter(params)
    
    def infer_masked_sequence_to_token_probs(
        self, 
        sequences: List[str], 
        variant_pos: int,
        variant_letters: List[str], 
        reference_letters: List[str],
        conditional_input=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Delegate to the base model's infer_masked_sequence_to_token_probs method.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            variant_pos: int - position of the variant in the input sequence (0-based)
            variant_letters: List[str] - variant nucleotide for each sequence
            reference_letters: List[str] - reference nucleotide for each sequence
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - variant_token_probs: probabilities for variant nucleotides (numpy array)
                - reference_token_probs: probabilities for reference nucleotides (numpy array)
        """
        return self.base_model.infer_masked_sequence_to_token_probs(
            sequences, variant_pos, variant_letters, reference_letters, conditional_input
        )
    
    def infer_variant_ref_sequences_to_labels_probs(
        self, 
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input=None
    ) -> Optional[np.ndarray]:
        """
        Forward pass through the model using variant and reference sequences.
        
        This method gets embeddings for both variant and reference sequences,
        combines them by concatenation, and applies the projection layer to get label probabilities.
        
        Args:
            variant_sequences: List[str] - batch of variant DNA sequences
            ref_sequences: List[str] - batch of reference DNA sequences
            conditional_input: Optional metadata inputs (not used in this wrapper)
            
        Returns:
            np.ndarray: Probabilities of shape [batch_size, num_labels] if projection layer exists,
                       otherwise returns None
        """
        # Get representative embeddings from base model using infer_sequence_to_sequence
        _, _, var_repr_np = self.base_model.infer_sequence_to_sequence(variant_sequences, conditional_input)
        _, _, ref_repr_np = self.base_model.infer_sequence_to_sequence(ref_sequences, conditional_input)
        
        if var_repr_np is None or ref_repr_np is None:
            return None
        
        # Apply projection layer if it exists
        if self.projection_layer is not None:
            with torch.no_grad():
                var_repr = torch.from_numpy(var_repr_np).to(self.device)
                ref_repr = torch.from_numpy(ref_repr_np).to(self.device)
                
                # Combine embeddings - concatenate variant and ref representations
                # Shape: [batch_size, hidden_dim * 2]
                combined_repr = torch.cat([var_repr, ref_repr], dim=1)
                
                logits = self.projection_layer(combined_repr)
                # Apply softmax to convert logits to probabilities
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
        else:
            return None


