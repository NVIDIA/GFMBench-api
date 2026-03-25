# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/armheb/DNA_bert_6 — Apache-2.0
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, BertForMaskedLM

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel


class DNABERTModel(BaseGFMModel):
    """
    Original DNABERT model with k-mer tokenization (k=6).
    Uses the pre-trained model from zhihan1996/DNABERT-6 which was used as a 
    baseline in the DNABERT-2 paper.
    
    Reference: https://arxiv.org/pdf/2306.15006
    Original Paper: Ji et al. "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome"
    """
    
    DEFAULT_KMER = 6
    # Note: Original DNABERT models are not on HuggingFace Hub
    # They need to be downloaded from: https://github.com/jerryji1993/DNABERT
    # Alternative: use "armheb/DNA_bert_6" which is a re-upload
    HUGGINGFACE_MODEL_NAME = "armheb/DNA_bert_6"
    
    def __init__(self, device='cpu', model_name=None, kmer=6, max_length=512):
        """
        Args:
            device: torch device ('cpu' or 'cuda')
            model_name: HuggingFace model identifier (default: zhihan1996/DNABERT-6)
            kmer: k-mer size for tokenization (default: 6)
            max_length: maximum sequence length (default: 512, BERT's max)
        """
        self.kmer = kmer
        self.device = device
        self.max_length = max_length
        
        # Use default model name if not provided
        if model_name is None:
            model_name = self.HUGGINGFACE_MODEL_NAME
        
        self.model_name = model_name
        
        # Load pre-trained tokenizer and model
        print(f"Loading DNABERT model (k-mer={kmer}): {model_name}")
        
        # Load from HuggingFace - use BertForMaskedLM to get both base model and prediction head
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = BertForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        
        # Move model to device
        self.model.to(device)
        
        # CLS token is at index 0 for BERT models
        self.cls_index = 0
        
        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"DNABERT loaded successfully. Hidden dim: {self.hidden_dim}, k-mer: {kmer}")
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.device = device
        return self
    
    def get_hidden_dim(self):
        """Return the hidden dimension of the model."""
        return self.hidden_dim
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def seq_to_kmers(self, seq, kmer=6):
        """
        Convert DNA sequence to k-mers with stride 1 (overlapping).
        
        Args:
            seq: DNA sequence string
            kmer: k-mer size
            
        Returns:
            list of k-mer strings
        """
        kmers = []
        for i in range(len(seq) - kmer + 1):
            kmers.append(seq[i:i+kmer])
        return kmers
    
    def tokenize_sequence(self, sequence):
        """
        Tokenize a single DNA sequence into k-mers and convert to token IDs.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            list of token IDs
        """
        # Convert to k-mers
        kmers = self.seq_to_kmers(sequence.upper(), self.kmer)
        
        # Join k-mers with spaces as expected by DNABERT tokenizer
        kmer_str = ' '.join(kmers)
        tokens = self.tokenizer.encode(kmer_str, add_special_tokens=True)
        
        return tokens
    
    def tokenize(self, sequences):
        """
        Tokenize DNA sequences using k-mer tokenization.
        
        Args:
            sequences: list of DNA strings
            
        Returns:
            dict with input_ids and attention_mask tensors
        """
        # Convert sequences to k-mer format
        kmer_sequences = []
        for seq in sequences:
            kmers = self.seq_to_kmers(seq.upper(), self.kmer)
            kmer_sequences.append(' '.join(kmers))
        
        # Tokenize
        encoded = self.tokenizer(
            kmer_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        encoded = {key: val.to(self.device) for key, val in encoded.items()}
        return encoded
    
    def _sequence_to_representative(self, sequences):
        """
        Internal method for training: get representative embeddings as torch tensors.
        
        Args:
            sequences: list of DNA strings
            
        Returns:
            torch.Tensor: representative embeddings of shape [batch, hidden_dim]
        """
        # Tokenize sequences
        encoded = self.tokenize(sequences)
        
        # Forward pass through model
        # BertForMaskedLM requires output_hidden_states=True to access hidden states
        outputs = self.model(**encoded, output_hidden_states=True)
        
        # Get sequence embeddings from hidden states: [batch, seq_len, hidden_dim]
        sequence_embeddings = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Extract CLS token embeddings (first token) as representative embeddings
        representative_embeddings = sequence_embeddings[:, self.cls_index, :]
        
        return representative_embeddings
    
    def infer_sequence_to_sequence_representative(self, sequences, conditional_input=None, require_grad: bool = False):
        """
        Forward pass through DNABERT model.
        
        Args:
            sequences: list of DNA strings
            conditional_input: Optional metadata inputs (not used in this model)
            require_grad: Whether to compute gradients (for finetuning backbone)
            
        Returns:
            tuple: (sequence_embeddings, representative_embeddings)
                - sequence_embeddings: torch.Tensor of shape [batch, seq_len, hidden_dim]
                - representative_embeddings: torch.Tensor of shape [batch, hidden_dim]
        """
        # Tokenize sequences
        encoded = self.tokenize(sequences)
        
        # Forward pass through model
        # BertForMaskedLM requires output_hidden_states=True to access hidden states
        outputs = self.model(**encoded, output_hidden_states=True)
        
        # Get sequence embeddings from hidden states: [batch, seq_len, hidden_dim]
        # For BertForMaskedLM, we need to get hidden_states from the output
        sequence_embeddings = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Extract CLS token embeddings (first token) as representative embeddings
        representative_embeddings = sequence_embeddings[:, self.cls_index, :]
        
        return sequence_embeddings, representative_embeddings
    
    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        """
        Get sequence-level outputs including embeddings and representative embeddings.
        For BERT models with bidirectional attention, all positions see full context.
        
        Args:
            sequences: list of str - batch of DNA sequences (e.g., ["ATCG", "GCTA"])
        
        Returns:
            tuple: (sequence_probs, sequence_embeddings, sequence_representative)
                - sequence_probs: None (not implemented - would require proper pseudo-log-likelihood
                                 with masked inference for each position, which is N times slower)
                - sequence_embeddings: np.ndarray [batch_size, seq_len, hidden_dim]
                                      Embeddings before vocab projection
                - sequence_representative: np.ndarray [batch_size, hidden_dim]
                                          CLS token embedding
        
        Note:
            For BERT models:
            - All positions have bidirectional context
            - sequence_representative uses the CLS token (position 0)
            - sequence_probs is None because single-pass unmasked inference does not yield
              proper pseudo-log-likelihood scores. True PLL requires masking each position
              and running N forward passes (see Salazar et al. 2020, "Masked Language Model Scoring")
        """
        # Tokenize sequences
        encoded = self.tokenize(sequences)
        
        # Forward pass through model to get hidden states
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
        
        # Get hidden states from the base model (last layer): [batch, seq_len, hidden_dim]
        # For BertForMaskedLM, hidden_states are in outputs.hidden_states
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Sequence representative is CLS token (position 0)
        sequence_representative = hidden_states[:, self.cls_index, :]  # [batch_size, hidden_dim]
        
        # Remove CLS and SEP tokens from hidden states
        hidden_states_seq = hidden_states[:, 1:-1, :]  # Remove CLS and SEP
        
        # Convert to numpy
        # sequence_probs is None - proper pseudo-log-likelihood requires N masked forward passes
        return (
            None,
            hidden_states_seq.cpu().numpy(),
            sequence_representative.cpu().numpy()
        )
    
    def sequence_pos_to_prob_pos(self, sequences, pos):
        """
        Map input DNA sequence position to output position for k-mer tokenization.
        
        For DNABERT with k=6 and stride=1, we return the k-mer index where the
        nucleotide at position `pos` appears at the center of the k-mer (position k//2).
        This follows BEND's approach for variant effect prediction.
        
        Since k-mer tokenization produces the same output positions for all sequences
        (independent of sequence content), we return an array with the same value for all sequences.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            pos: int - position in the input DNA sequence (0-based)
        
        Returns:
            np.ndarray: array of shape [batch_size] with k-mer token indices where the nucleotide is at position k//2 (center)
        
        Example:
            For k=6, input ["ATCGATCG", "GCTAGCTA"]:
            - K-mers: ["ATCGAT"(0), "TCGATC"(1), "CGATCG"(2)]
            - pos=2 ('C'): appears at position 2 in k-mer 0 → output_pos=[0, 0]
            - pos=3 ('G'): appears at position 2 in k-mer 1 → output_pos=[1, 1]
            - pos=5 ('T'): appears at position 2 in k-mer 3 → output_pos=[3, 3]
            
            Generally: k-mer starting at position (pos - k//2) has the nucleotide at center
        """
        # For a nucleotide at position `pos` to be at the center (position k//2) of a k-mer,
        # that k-mer must start at position (pos - k//2)
        kmer_center = self.kmer // 2
        output_pos = pos - kmer_center
        
        # K-mer tokenization is content-independent, so all sequences have the same output position
        batch_size = len(sequences)
        return np.full(batch_size, output_pos, dtype=np.int32)
    
    def infer_sequence_to_labels_probs(self, sequences, conditional_input=None):
        """Not implemented for this model."""
        return None
    
    def infer_variant_ref_sequences_to_labels_probs(self, variant_sequences, ref_sequences, conditional_input=None):
        """Not implemented for this model."""
        return None
    
    def infer_masked_sequence_to_token_probs(
        self, 
        sequences: List[str], 
        variant_pos: int,
        variant_letters: List[str], 
        reference_letters: List[str],
        conditional_input=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get token probabilities at a specific position using masked prediction for DNABERT.
        
        For DNABERT's k-mer tokenization, we mask the k-mer that contains the variant position
        at its center and predict probabilities for the variant and reference k-mers.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            variant_pos: int - position of the variant in the input sequence (0-based)
            variant_letters: List[str] - variant nucleotide for each sequence
            reference_letters: List[str] - reference nucleotide for each sequence
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - variant_token_probs: Array of shape [batch_size] with probabilities for variant k-mers
                - reference_token_probs: Array of shape [batch_size] with probabilities for reference k-mers
                - Returns (None, None) if model doesn't have MLM head
        """
        # BertForMaskedLM always supports masked prediction
        
        batch_size = len(sequences)
        
        # Use sequence_pos_to_prob_pos to find which k-mer contains the variant at its center
        # This ensures consistency with how we map positions elsewhere
        kmer_positions = self.sequence_pos_to_prob_pos(sequences, variant_pos)
        kmer_start_pos = int(kmer_positions[0])  # Same for all sequences (content-independent for k-mer tokenization)
        
        # Calculate k-mer center for later use in constructing variant/reference k-mers
        kmer_center = self.kmer // 2
        
        # Validate that the position is valid for k-mer extraction
        if kmer_start_pos < 0 or (kmer_start_pos + self.kmer) > len(sequences[0]):
            # Position is too close to sequence boundaries for proper k-mer
            return None, None
        
        # Convert sequences to k-mers and mask the k-mer containing the variant
        masked_kmer_sequences = []
        for seq in sequences:
            # Create k-mers from the original sequence
            kmers = self.seq_to_kmers(seq.upper(), self.kmer)
            # Replace the k-mer at the variant position with [MASK]
            # The k-mer at index kmer_start_pos contains the nucleotide at variant_pos
            kmers[kmer_start_pos] = '[MASK]'
            masked_kmer_sequences.append(' '.join(kmers))
        
        # Tokenize masked sequences
        encoded = self.tokenizer(
            masked_kmer_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        encoded = {key: val.to(self.device) for key, val in encoded.items()}
        
        # Find the position of [MASK] token in tokenized sequence
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_positions) != batch_size:
            # Something went wrong with masking
            return None, None
        
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = self.model(**encoded)
            # BertForMaskedLM directly outputs logits
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Extract probabilities at masked positions
        variant_probs_list = []
        reference_probs_list = []
        
        for i in range(batch_size):
            mask_pos = mask_positions[i]
            
            # Get the k-mer tokens for variant and reference
            # Build the k-mers with variant and reference nucleotides
            seq = sequences[i]
            
            # Create variant and reference k-mers
            variant_kmer = seq[kmer_start_pos:kmer_start_pos + kmer_center] + variant_letters[i] + seq[kmer_start_pos + kmer_center + 1:kmer_start_pos + self.kmer]
            reference_kmer = seq[kmer_start_pos:kmer_start_pos + kmer_center] + reference_letters[i] + seq[kmer_start_pos + kmer_center + 1:kmer_start_pos + self.kmer]
            
            # Get token IDs for these k-mers
            variant_token_id = self.tokenizer.encode(variant_kmer, add_special_tokens=False)[0]
            reference_token_id = self.tokenizer.encode(reference_kmer, add_special_tokens=False)[0]
            
            # Extract probabilities for variant and reference tokens
            variant_prob = probs[i, mask_pos, variant_token_id].item()
            reference_prob = probs[i, mask_pos, reference_token_id].item()
            
            variant_probs_list.append(variant_prob)
            reference_probs_list.append(reference_prob)
        
        # Convert to numpy arrays
        variant_token_probs = np.array(variant_probs_list, dtype=np.float32)
        reference_token_probs = np.array(reference_probs_list, dtype=np.float32)
        
        return variant_token_probs, reference_token_probs
    
    def get_hidden_dim(self):
        """
        Get the hidden dimension size of the model.
        
        Returns:
            int: hidden dimension size
        """
        return self.hidden_dim
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a fine-tuned checkpoint for the model.
        
        Args:
            checkpoint_path: path to checkpoint file
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)
        
        print("Checkpoint loaded successfully")

