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

# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/zhihan1996/DNABERT-2-117M — Apache-2.0
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel


@dataclass
class MLMOutput:
    """Output container for MLM model, compatible with HuggingFace format."""
    loss: torch.Tensor
    logits: torch.Tensor


class MLMModelWrapper(nn.Module):
    """
    Wrapper that combines encoder + MLM head for training.
    
    Behaves like AutoModelForMaskedLM:
    - Accepts (input_ids, attention_mask, labels)
    - Returns object with .loss and .logits attributes
    - Computes cross-entropy loss for MLM training
    """
    
    def __init__(self, encoder: nn.Module, mlm_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = mlm_head
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> MLMOutput:
        """
        Forward pass for MLM training.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for MLM [batch, seq_len], -100 for non-masked
        
        Returns:
            MLMOutput with loss and logits
        """
        # Forward through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
        
        # Forward through MLM head
        logits = self.mlm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return MLMOutput(loss=loss, logits=logits)
    
    def parameters(self):
        """Return all parameters (encoder + MLM head)."""
        import itertools
        return itertools.chain(self.encoder.parameters(), self.mlm_head.parameters())
    
    def train(self, mode=True):
        """Set training mode."""
        self.encoder.train(mode)
        self.mlm_head.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.encoder.eval()
        self.mlm_head.eval()
        return self


class DNABERT2Model(BaseGFMModel):
    """
    DNA-BERT2 pre-trained model for genomic sequence analysis.
    Uses the 117M parameter model from zhihan1996/DNABERT-2-117M
    which achieved state-of-the-art results on promoter detection.
    
    Architecture:
        - self.model: The encoder (transformer) - used for embeddings
        - self.mlm_head: The MLM prediction head - used for masked token prediction
    
    Reference: https://huggingface.co/zhihan1996/DNABERT-2-117M
    Paper: Zhou et al. "DNABERT-2: Efficient Foundation Model for Multi-Species Genome"
    """
    
    def __init__(self, device='cpu', model_name="zhihan1996/DNABERT-2-117M", max_length=512, pretrained=True):
        """
        Args:
            device: torch device ('cpu' or 'cuda')
            model_name: HuggingFace model identifier for DNABERT-2
            max_length: int - maximum sequence length for tokenization (default: 512)
                        DNABERT-2 was trained on 512 tokens but can extrapolate to longer
                        sequences thanks to ALiBi positional encoding.
            pretrained: bool - if True (default), load pre-trained weights from HuggingFace.
                        If False, initialize model with random weights (for training from scratch).
        
        Note:
            On CUDA devices, DNABERT-2 uses flash attention (via the flash-attn package) 
            for faster inference. The cached model code has been patched to use the 
            flash-attn package instead of the original triton implementation.
            Requires: conda activate sense-env-att
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pretrained = pretrained
        
        # Disable flash attention for CPU - must be set BEFORE model loads
        import os
        if device == 'cpu':
            os.environ['FLASH_ATTN_SKIP_CUDA_BUILD'] = '1'
            os.environ['DNABERT2_DISABLE_FLASH'] = '1'  # Custom flag for patched bert_layers
        
        # Load tokenizer (always from pre-trained, needed for vocab)
        print(f"Loading DNA-BERT2 model: {model_name} (pretrained={pretrained})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load full MLM model, either pre-trained or from scratch
        # For CPU, use eager attention implementation to avoid flash_attn
        load_kwargs = {"trust_remote_code": True}
        if device == 'cpu':
            load_kwargs["attn_implementation"] = "eager"
        
        if pretrained:
            # Load pre-trained weights from HuggingFace
            mlm_full = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
        else:
            # Initialize with random weights (train from scratch)
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            mlm_full = AutoModelForMaskedLM.from_config(config, **load_kwargs)
            print("  -> Initialized with random weights (training from scratch)")
        
        # Extract encoder (self.model) and MLM head separately
        self.model = mlm_full.bert          # The encoder/transformer
        self.mlm_head = mlm_full.cls        # The MLM prediction head
        
        # Clean up the wrapper (encoder and head are now separate references)
        del mlm_full
        
        # Move to device
        self.model.to(device)
        self.mlm_head.to(device)
        
        # CLS token is at index 0 for DNABERT-2
        self.cls_index = 0
        
        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size
        
        # Track whether MLM head has valid weights (True by default since we load from HuggingFace)
        self.mlm_head_loaded = True
        
        # Handle flash attention - force disable on CPU
        import sys
        attn_module = self.model.encoder.layer[0].attention.self.__class__.__module__
        bert_layers = sys.modules.get(attn_module)
        module_file = bert_layers.__file__ if bert_layers else "unknown"
        
        # Force disable flash attention on CPU by patching the module
        # The bert_layers.py checks `if not FLASH_ATTN_AVAILABLE:` to use PyTorch fallback
        if device == 'cpu' and bert_layers is not None:
            bert_layers.FLASH_ATTN_AVAILABLE = False
            bert_layers.flash_attn_qkvpacked_func = None  # Also nullify the function
            print("  -> Flash attention disabled for CPU")
        
        flash_available = getattr(bert_layers, 'FLASH_ATTN_AVAILABLE', None) if bert_layers else None
        
        if device == 'cpu':
            attn_type = "PyTorch attention (CPU)"
        elif flash_available is True:
            attn_type = "✓ FLASH ATTENTION ENABLED"
        elif flash_available is False:
            attn_type = "PyTorch attention (flash-attn unavailable)"
        elif flash_available is None:
            attn_type = "✗ UNPATCHED bert_layers.py - may use triton"
        else:
            attn_type = "unknown attention mode"
        
        # Check if using /efs cache
        hf_home = os.environ.get('HF_HOME', 'not set')
        cache_status = "✓ /efs cache" if '/efs/' in module_file else f"✗ NOT /efs (HF_HOME={hf_home})"
        
        print(f"DNA-BERT2 loaded successfully. Hidden dim: {self.hidden_dim}, max_length: {self.max_length}")
        print(f"  Attention: {attn_type}")
        print(f"  Cache: {cache_status}")
        print(f"  Module: {module_file}")
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        self.mlm_head.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode."""
        self.model.train(mode)
        self.mlm_head.train(mode)
        return self
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.mlm_head.to(device)
        self.device = device
        return self
    
    def get_hidden_dim(self):
        """Return the hidden dimension of the model."""
        return self.hidden_dim
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def tokenize(self, sequences):
        """
        Tokenize DNA sequences using DNABERT-2 tokenizer.
        
        Args:
            sequences: list of DNA strings
            
        Returns:
            dict with input_ids and attention_mask tensors
        """
        # DNABERT-2 tokenizer handles DNA sequences directly
        encoded = self.tokenizer(
            sequences,
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
        # DNABERT-2's AutoModel returns a tuple: (sequence_embeddings, pooler_output)
        outputs = self.model(**encoded)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            # Returns tuple: (sequence_embeddings, pooler_output)
            sequence_embeddings = outputs[0]
        else:
            # Returns object with last_hidden_state attribute
            sequence_embeddings = outputs.last_hidden_state
        
        # Extract CLS token embeddings (first token) as representative embeddings
        representative_embeddings = sequence_embeddings[:, self.cls_index, :]
        
        return representative_embeddings
    
    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        """
        Get sequence-level outputs including embeddings and representative embeddings.
        For DNABERT-2 with bidirectional attention, all positions see full context.
        
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
            For DNABERT-2:
            - All positions have bidirectional context
            - sequence_representative uses the CLS token (position 0)
            - sequence_probs is None because single-pass unmasked inference does not yield
              proper pseudo-log-likelihood scores. True PLL requires masking each position
              and running N forward passes (see Salazar et al. 2020, "Masked Language Model Scoring")
        """
        # Tokenize sequences
        encoded = self.tokenize(sequences)
        
        # Forward pass through encoder
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs.last_hidden_state
        
        # Sequence representative is CLS token (position 0)
        sequence_representative = hidden_states[:, self.cls_index, :]  # [batch_size, hidden_dim]
        
        # Remove CLS and SEP tokens from hidden states
        hidden_states_seq = hidden_states[:, 1:-1, :]
        
        # Convert to numpy (detach in case we're in training mode)
        # sequence_probs is None - proper pseudo-log-likelihood requires N masked forward passes
        return (
            None,
            hidden_states_seq.detach().cpu().numpy(),
            sequence_representative.detach().cpu().numpy()
        )
    
    def sequence_pos_to_prob_pos(self, sequences, pos):
        """
        Map input DNA sequence position to output position for DNABERT-2.
        
        DNABERT-2 uses BPE tokenization, which means the output position depends on
        how each specific sequence is tokenized. This method analyzes each sequence
        to find which token contains the nucleotide at position `pos`.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            pos: int - position in the input DNA sequence (0-based)
        
        Returns:
            np.ndarray: array of shape [batch_size] with output positions for each sequence
                       Returns -1 for sequences where the position cannot be determined
        
        Note:
            Since BPE tokenization is sequence-dependent, different sequences may have
            different output positions even for the same input position.
        """
        batch_size = len(sequences)
        output_positions = np.zeros(batch_size, dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            try:
                # Tokenize the sequence
                encoded = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
                input_ids = encoded['input_ids'][0].tolist()
                
                # Get the character span for each token
                # Note: offset_mapping tells us which characters each token corresponds to
                encoded_with_offsets = self.tokenizer(seq, return_offsets_mapping=True, add_special_tokens=True)
                offset_mapping = encoded_with_offsets['offset_mapping']
                
                # Find which token contains position `pos`
                # offset_mapping is a list of (start, end) tuples for each token
                token_idx = -1
                for idx, (start, end) in enumerate(offset_mapping):
                    if start <= pos < end:
                        token_idx = idx
                        break
                
                # Account for special tokens (CLS is at position 0, need to remove it from output position)
                if token_idx > 0:  # token_idx=0 is CLS, actual sequence tokens start at 1
                    output_positions[i] = token_idx - 1  # Subtract 1 to account for CLS token removal
                else:
                    output_positions[i] = -1  # Invalid position
                    
            except Exception as e:
                # If tokenization fails, mark as invalid
                output_positions[i] = -1
        
        return output_positions
    
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
        Get token probabilities at a specific position using masked prediction for DNABERT-2.
        
        For DNABERT-2's BPE tokenization, we mask the token containing the variant position
        and predict probabilities. Since BPE tokens may span multiple nucleotides, we find
        the tokens that would result in the variant vs reference nucleotide at that position.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            variant_pos: int - position of the variant in the input sequence (0-based)
            variant_letters: List[str] - variant nucleotide for each sequence
            reference_letters: List[str] - reference nucleotide for each sequence
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - variant_token_probs: Array of shape [batch_size] with probabilities for variant, or None if MLM head not loaded
                - reference_token_probs: Array of shape [batch_size] with probabilities for reference, or None if MLM head not loaded
        
        Note:
            Returns (None, None) if mlm_head_loaded=False since this method requires the MLM head.
        """
        # Check if MLM head is available
        if not self.mlm_head_loaded:
            return None, None
        
        batch_size = len(sequences)
        
        # Create masked sequences - replace nucleotide at variant_pos with [MASK] token
        masked_sequences = []
        for seq in sequences:
            # Create masked sequence by replacing the nucleotide at variant_pos
            masked_seq = seq[:variant_pos] + self.tokenizer.mask_token + seq[variant_pos + 1:]
            masked_sequences.append(masked_seq)
        
        # Tokenize masked sequences
        encoded = self.tokenizer(
            masked_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        encoded = {key: val.to(self.device) for key, val in encoded.items()}
        
        # Find positions of [MASK] tokens
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (encoded['input_ids'] == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) != batch_size:
            # Masking didn't work as expected
            return None, None
        
        # Forward pass through encoder + MLM head to get predictions
        with torch.no_grad():
            outputs = self.model(**encoded)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs.last_hidden_state
            
            logits = self.mlm_head(hidden_states)  # [batch, seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Get token IDs for single nucleotides
        # DNABERT-2 should have tokens for A, T, C, G
        nucleotide_token_ids = {}
        for nuc in ['A', 'T', 'C', 'G']:
            # Encode single nucleotide to get its token ID
            tokens = self.tokenizer.encode(nuc, add_special_tokens=False)
            if len(tokens) == 1:
                nucleotide_token_ids[nuc] = tokens[0]
            else:
                # Fallback: try lowercase
                tokens = self.tokenizer.encode(nuc.lower(), add_special_tokens=False)
                if len(tokens) == 1:
                    nucleotide_token_ids[nuc] = tokens[0]
        
        if len(nucleotide_token_ids) < 4:
            # Can't find single nucleotide tokens
            return None, None
        
        # Extract probabilities for variant and reference nucleotides
        variant_probs_list = []
        reference_probs_list = []
        
        for i in range(batch_size):
            # Get the mask position for this sequence
            batch_mask_indices = (mask_positions[0] == i).nonzero(as_tuple=True)[0]
            if len(batch_mask_indices) == 0:
                variant_probs_list.append(0.0)
                reference_probs_list.append(0.0)
                continue
            
            mask_pos = mask_positions[1][batch_mask_indices[0]]
            
            # Get variant and reference nucleotides
            var_nuc = variant_letters[i].upper()
            ref_nuc = reference_letters[i].upper()
            
            # Get token IDs
            var_token_id = nucleotide_token_ids.get(var_nuc)
            ref_token_id = nucleotide_token_ids.get(ref_nuc)
            
            if var_token_id is None or ref_token_id is None:
                variant_probs_list.append(0.0)
                reference_probs_list.append(0.0)
                continue
            
            # Extract probabilities
            variant_prob = probs[i, mask_pos, var_token_id].item()
            reference_prob = probs[i, mask_pos, ref_token_id].item()
            
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
    
    def load_checkpoint(self, checkpoint_path: str, load_mlm_head: bool = True):
        """
        Load a checkpoint for the model.
        
        Handles different checkpoint formats:
        - MLM checkpoint (mlm_model_state_dict): Load encoder + MLM head weights
        - Base checkpoint (model_state_dict): Load only encoder weights
        - MLM head checkpoint (mlm_head_state_dict): Load MLM head weights
        
        Args:
            checkpoint_path: path to checkpoint file (.pt)
            load_mlm_head: if True and MLM head weights available, load them
        
        Note:
            Sets self.mlm_head_loaded = True if MLM head weights were loaded from checkpoint,
            False otherwise. Methods that require MLM head will return None when mlm_head_loaded=False.
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        
        # Track whether we loaded MLM head weights from the checkpoint
        mlm_head_was_loaded = False
        
        # Handle different checkpoint formats
        if 'mlm_model_state_dict' in state and load_mlm_head:
            # =================================================================
            # Legacy MLM checkpoint format (full AutoModelForMaskedLM state)
            # Extract encoder (bert.*) and MLM head (cls.*) weights
            # =================================================================
            mlm_state = state['mlm_model_state_dict']
            
            # Extract encoder weights (remove 'bert.' prefix)
            encoder_state = {}
            for key, value in mlm_state.items():
                if key.startswith('bert.'):
                    encoder_state[key[5:]] = value  # Remove 'bert.' prefix
            
            # Extract MLM head weights (remove 'cls.' prefix)
            mlm_head_state = {}
            for key, value in mlm_state.items():
                if key.startswith('cls.'):
                    mlm_head_state[key[4:]] = value  # Remove 'cls.' prefix
            
            # Load weights
            self.model.load_state_dict(encoder_state, strict=False)
            self.mlm_head.load_state_dict(mlm_head_state, strict=False)
            mlm_head_was_loaded = True
            print("Loaded encoder and MLM head from legacy MLM checkpoint")
            
        elif 'model_state_dict' in state:
            # =================================================================
            # Base model checkpoint (e.g., from JEPA or new format)
            # Load encoder weights only, MLM head keeps original weights
            # Use strict=False to handle pooler layer differences
            # =================================================================
            self.model.load_state_dict(state['model_state_dict'], strict=False)
            print("Loaded encoder weights")
            
            # Optionally load MLM head if available in new format
            if 'mlm_head_state_dict' in state and load_mlm_head:
                self.mlm_head.load_state_dict(state['mlm_head_state_dict'])
                mlm_head_was_loaded = True
                print("Loaded MLM head weights")
            else:
                mlm_head_was_loaded = False
                print("  -> MLM head NOT loaded (checkpoint missing mlm_head_state_dict)")
            
        else:
            # Assume it's just the encoder state dict directly
            # Use strict=False to handle pooler layer differences
            self.model.load_state_dict(state, strict=False)
            mlm_head_was_loaded = False
            print("Loaded encoder weights (raw state dict, no MLM head)")
        
        # Update the mlm_head_loaded attribute
        self.mlm_head_loaded = mlm_head_was_loaded
        
        print(f"Checkpoint loaded successfully (mlm_head_loaded={self.mlm_head_loaded})")
    
    def save_checkpoint(self, checkpoint_path: str, save_mlm_head: bool = True, extra_state: dict = None):
        """
        Save a checkpoint for the model.
        
        Args:
            checkpoint_path: path to save checkpoint (.pt)
            save_mlm_head: if True, also save MLM head weights
            extra_state: additional state to save (e.g., optimizer, epoch)
        """
        import os
        os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
        }
        
        if save_mlm_head:
            state['mlm_head_state_dict'] = self.mlm_head.state_dict()
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    def get_mlm_head(self):
        """
        Get the MLM prediction head.
        
        Returns:
            nn.Module: the MLM prediction head (BertOnlyMLMHead)
        """
        return self.mlm_head
    
    def get_mlm_model(self):
        """
        Get a combined model (encoder + MLM head) for MLM training.
        
        Returns a wrapper that behaves like AutoModelForMaskedLM:
        - Accepts (input_ids, attention_mask, labels)
        - Returns object with .loss and .logits attributes
        
        Returns:
            MLMModelWrapper: Combined model for MLM training
        """
        return MLMModelWrapper(self.model, self.mlm_head)
    
    def get_tokenizer(self):
        """
        Get the tokenizer.
        
        Returns:
            tokenizer: the model's tokenizer
        """
        return self.tokenizer
    
    def load_pretrained_embeddings(self, model_name: str = None):
        """
        Load pretrained token embeddings into current model.
        
        This is useful when training from scratch (pretrained=False) but wanting
        to use pretrained token embeddings (e.g., to prevent mode collapse).
        
        Args:
            model_name: HuggingFace model identifier. If None, uses self.model_name.
        """
        model_name = model_name or self.model_name
        print(f"Loading pretrained token embeddings from: {model_name}")
        
        # Load pretrained model temporarily
        pretrained = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        
        # Copy word embeddings from pretrained to current model
        self.model.embeddings.word_embeddings.load_state_dict(
            pretrained.bert.embeddings.word_embeddings.state_dict()
        )
        
        # Clean up pretrained model
        del pretrained
        print("  -> Pretrained token embeddings loaded successfully")
    
    def freeze_token_embeddings(self):
        """
        Freeze the token embeddings layer (word_embeddings).
        
        Sets requires_grad=False for all parameters in the word embeddings layer.
        """
        for param in self.model.embeddings.word_embeddings.parameters():
            param.requires_grad = False
        print("  -> Token embeddings frozen (requires_grad=False)")
    
    def mlm_head_parameters(self):
        """
        Get parameters of the MLM head only (for training MLM head separately).
        
        Returns:
            Iterator of MLM head parameters
        """
        return self.mlm_head.parameters()
    
    def all_parameters(self):
        """
        Get all parameters (encoder + MLM head).
        
        Returns:
            Iterator of all model parameters
        """
        import itertools
        return itertools.chain(self.model.parameters(), self.mlm_head.parameters())
    
    def parameters(self):
        """
        Get encoder parameters (nn.Module compatible interface).
        
        Returns:
            Iterator of encoder parameters
        """
        return self.model.parameters()
    
    def named_parameters(self, prefix='', recurse=True):
        """
        Get named encoder parameters (nn.Module compatible interface).
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: Whether to include parameters of submodules
        
        Returns:
            Iterator of (name, parameter) tuples
        """
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

