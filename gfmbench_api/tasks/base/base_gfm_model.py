# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

class BaseGFMModel(ABC):
    """
    Base class for Genomic Foundation Models.
    Concrete classes must implement abstract methods.
    """
    
    @abstractmethod
    def __init__(self, device: str = 'cpu') -> None:
        """
        Initialize the model.
        
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        pass
    
    @abstractmethod
    def infer_sequence_to_labels_probs(
        self, 
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Get label probabilities for sequences (for classification tasks).
        
        Args:
            sequences: list of str - batch of DNA sequences (e.g., ["ATCG", "GCTA"])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            np.ndarray of shape [batch_size, num_labels] OR None
            Probabilities for each label. Returns None if not implemented.
        """
        pass
    
    @abstractmethod
    def infer_sequence_to_sequence(
        self, 
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get sequence-level outputs for zero-shot benchmarks including probabilities,
        embeddings, and representative embeddings.
        
        Args:
            sequences: list of str - batch of DNA sequences (e.g., ["ATCG", "GCTA"])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            tuple: (sequence_probs, sequence_embeddings, sequence_representative)
                - sequence_probs: np.ndarray of shape [batch_size, seq_len] OR None
                                 Probability for each nucleotide in each sequence.
                                 For input ["ATCT", "ATCG"], returns:
                                 [[P(A), P(T), P(C), P(T)], [P(A), P(T), P(C), P(G)]]
                                 Returns None if model doesn't have vocabulary projection head (e.g., BERT without MLM head)
                                 
                - sequence_embeddings: np.ndarray of shape [batch_size, seq_len, hidden_dim] OR None
                                      Embeddings (logits before vocabulary projection) for each position.
                                      Returns None if model doesn't support this output.
                                      
                - sequence_representative: np.ndarray of shape [batch_size, hidden_dim] OR None
                                          Single embedding representing the entire sequence.
                                          For BERT: CLS token embedding
                                          For GPT: Last token embedding
                                          Returns None if model doesn't support this output.
        
        Note:
            - This method is used for two prediction approaches:
              1. LLR calculation using sequence_probs (Method 1)
              2. Cosine similarity calculation using sequence_representative (Method 2, BEND-style)
            - Models can return None for outputs they don't support
            - The benchmark will skip calculating metrics for unsupported outputs
            - At least one output should be non-None for the method to be useful
        """
        pass
    
    @abstractmethod
    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        """
        Map input DNA sequence position to output sequence position (accounting for tokenization).
        
        This is crucial for models with different tokenization strategies to correctly
        identify which output position corresponds to a given input nucleotide position.
        For tokenizers like BPE where the output position depends on the sequence content,
        this method computes the mapping for each sequence in the batch.
        
        Args:
            sequences: List[str] - batch of DNA sequences to analyze
                                  For BPE tokenizers, the tokenization depends on sequence content
            pos: int - position in the input DNA sequence (0-based index)
                     For sequence "ATCGTA", pos=0 is 'A', pos=3 is 'G', etc.
        
        Returns:
            np.ndarray: array of shape [batch_size] containing the corresponding output position
                       for each sequence. This is the index in sequence_embeddings/sequence_probs
                       that corresponds to the prediction for the nucleotide at input position `pos`.
                       Returns -1 for sequences where the position cannot be determined.
        
        Examples:
            - Single nucleotide tokenizer (e.g., SimpleDNATransformer):
              Input: ["ATCG", "GCTA"] → Tokens: [[CLS, A, T, C, G], [CLS, G, C, T, A]]
              pos=0 ('A'/'G') → output_pos=[0, 0] (after removing CLS, same for all sequences)
              
            - Autoregressive model (e.g., GPT):
              Input: ["ATCG", "GCTA"] → Output predicts next token at each position
              pos=1 ('T'/'C') → output_pos=[0, 0] (prediction at position 0 for token at 1)
              
            - K-mer tokenizer (e.g., 6-mer DNABERT):
              Input: ["ATCGATCG", "GCTAGCTA"] → 6-mers vary by sequence
              pos=2 → output_pos=[0, 0] (both in first k-mer)
              
            - BPE tokenizer (sequence-dependent):
              Input: ["ATCGATCG", "AAAAAACG"] → tokens depend on learned vocabulary
              pos=2 → output_pos=[1, 0] (can vary based on how each sequence tokenizes)
        
        Note:
            - This method accounts for special tokens (CLS, SEP) being removed from outputs
            - For k-mer models, returns the token index where the nucleotide is at position k//2
            - For autoregressive models, returns pos-1 (since output at i predicts token at i+1)
            - For BPE and other content-dependent tokenizers, analyzes each sequence separately
            - Returns -1 for any sequence where the position cannot be determined
        """
        pass
    
    @abstractmethod
    def infer_masked_sequence_to_token_probs(
        self, 
        sequences: List[str], 
        variant_pos: int,
        variant_letters: List[str], 
        reference_letters: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get token probabilities at a specific position using masked prediction.
        
        This method is used for SNV variant effect prediction where we mask the variant
        position and predict probabilities for both the variant and reference nucleotides.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            variant_pos: int - position of the variant in the input sequence (0-based)
            variant_letters: List[str] - variant nucleotide for each sequence (e.g., ['A', 'T', 'G'])
            reference_letters: List[str] - reference nucleotide for each sequence (e.g., ['G', 'C', 'A'])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - variant_token_probs: Array of shape [batch_size] containing the probability
                                      of the variant nucleotide at the masked position for each sequence
                - reference_token_probs: Array of shape [batch_size] containing the probability
                                        of the reference nucleotide at the masked position for each sequence
                - Returns (None, None) if model doesn't support masked prediction
        
        Implementation notes:
            - Mask the sequences at variant_pos (e.g., replace with [MASK] token or similar)
            - Run model inference to predict the masked token
            - Extract probability for variant_letters[i] and reference_letters[i] for each sequence i
            - This method can leverage existing methods like tokenization and inference
            - Return (None, None) for models that don't support masked prediction
        
        Example:
            sequences = ["ATCG", "GCTA"]
            variant_pos = 1
            variant_letters = ["G", "A"]  # New nucleotides
            reference_letters = ["T", "C"]  # Original nucleotides
            
            # Masks position 1: ["A[MASK]CG", "G[MASK]TA"]
            # Returns probabilities: (P(G), P(A)), (P(T), P(C))
        """
        pass
    
    @abstractmethod
    def infer_variant_ref_sequences_to_labels_probs(
        self, 
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Get label probabilities for variant/reference sequence pairs.
        
        This method is used for variant effect prediction tasks where the model
        receives both variant and reference sequences as input and predicts
        classification labels.
        
        Args:
            variant_sequences: List[str] - batch of variant DNA sequences
            ref_sequences: List[str] - batch of reference DNA sequences
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            np.ndarray of shape [batch_size, num_labels] OR None
            Probabilities for each label. Returns None if not implemented.
        """
        pass