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

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


class DNABERT2Model(nn.Module):
    """zhihan1996/DNABERT-2-117M as AutoModelForMaskedLM (encoder + MLM head) for benchmark infer_* APIs."""

    def __init__(
        self,
        device="cpu",
        model_name="zhihan1996/DNABERT-2-117M",
        max_length=512,
        pretrained=True,
        use_flash_attention=False,
    ):
        """
        Args:
            device: torch device ('cpu' or 'cuda')
            model_name: HuggingFace model identifier for DNABERT-2
            max_length: maximum sequence length for tokenization (default: 512)
            pretrained: if True (default), load pre-trained weights from HuggingFace
            use_flash_attention: if False (default), use eager attention for deterministic eval
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pretrained = pretrained

        import os

        if device == "cpu":
            os.environ["FLASH_ATTN_SKIP_CUDA_BUILD"] = "1"
            os.environ["DNABERT2_DISABLE_FLASH"] = "1"

        print(f"Loading DNA-BERT2 model: {model_name} (pretrained={pretrained})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if device == "cuda" and not use_flash_attention:
            import sys

            for module_name, module in list(sys.modules.items()):
                if "bert_layers" in module_name and hasattr(module, "FLASH_ATTN_AVAILABLE"):
                    module.FLASH_ATTN_AVAILABLE = False
                    if hasattr(module, "flash_attn_qkvpacked_func"):
                        module.flash_attn_qkvpacked_func = None
                    print(f"  -> Pre-patched {module_name} to disable flash attention")
                    break

        load_kwargs = {"trust_remote_code": True}
        if device == "cpu" or not use_flash_attention:
            load_kwargs["attn_implementation"] = "eager"

        if pretrained:
            mlm_full = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            mlm_full = AutoModelForMaskedLM.from_config(config, **load_kwargs)
            print("  -> Initialized with random weights (training from scratch)")

        self.model = mlm_full
        self.add_module("model", mlm_full)
        self.model.to(device)

        self.cls_index = 0
        self.hidden_dim = self.model.bert.config.hidden_size
        self.mlm_head_loaded = True

        import sys

        attn_module = self.model.bert.encoder.layer[0].attention.self.__class__.__module__
        bert_layers = sys.modules.get(attn_module)
        module_file = bert_layers.__file__ if bert_layers else "unknown"

        if bert_layers is not None:
            if device == "cpu":
                bert_layers.FLASH_ATTN_AVAILABLE = False
                bert_layers.flash_attn_qkvpacked_func = None
                print("  -> Flash attention disabled for CPU (using eager attention)")
            elif not use_flash_attention:
                bert_layers.FLASH_ATTN_AVAILABLE = False
                bert_layers.flash_attn_qkvpacked_func = None
                print("  -> Flash attention disabled for CUDA (using eager attention)")

        flash_available = getattr(bert_layers, "FLASH_ATTN_AVAILABLE", None) if bert_layers else None

        if device == "cpu":
            attn_type = "PyTorch attention (CPU)"
        elif flash_available is True:
            attn_type = "✓ FLASH ATTENTION ENABLED"
        elif flash_available is False:
            attn_type = "PyTorch attention (flash-attn unavailable/disabled)"
        elif flash_available is None:
            attn_type = "✗ UNPATCHED bert_layers.py - may use triton"
        else:
            attn_type = "unknown attention mode"

        print(f"DNA-BERT2 loaded successfully. Hidden dim: {self.hidden_dim}, max_length: {self.max_length}")
        print(f"  Attention: {attn_type}")
        print(f"  Module: {module_file}")

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def get_hidden_dim(self):
        return self.hidden_dim

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in self.model.named_parameters(prefix="", recurse=recurse):
            yield name, param

    def state_dict(self, prefix: str = "", keep_vars: bool = False):
        return self.model.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def tokenize(self, sequences):
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {key: val.to(self.device) for key, val in encoded.items()}

    def _sequence_to_representative(self, sequences):
        encoded = self.tokenize(sequences)
        outputs = self.model.bert(**encoded)
        sequence_embeddings = outputs[0]
        return sequence_embeddings[:, self.cls_index, :]

    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        encoded = self.tokenize(sequences)

        with torch.no_grad():
            outputs = self.model.bert(**encoded)
        hidden_states = outputs[0]

        sequence_representative = hidden_states[:, self.cls_index, :]
        hidden_states_seq = hidden_states[:, 1:-1, :]

        return (
            None,
            hidden_states_seq.detach().cpu().numpy(),
            sequence_representative.detach().cpu().numpy(),
        )

    def sequence_pos_to_prob_pos(self, sequences, pos, conditional_input=None):
        batch_size = len(sequences)
        output_positions = np.zeros(batch_size, dtype=np.int32)

        for i, seq in enumerate(sequences):
            try:
                encoded_with_offsets = self.tokenizer(
                    seq, return_offsets_mapping=True, add_special_tokens=True
                )
                offset_mapping = encoded_with_offsets["offset_mapping"]

                token_idx = -1
                for idx, (start, end) in enumerate(offset_mapping):
                    if start <= pos < end:
                        token_idx = idx
                        break

                if token_idx > 0:
                    output_positions[i] = token_idx - 1
                else:
                    output_positions[i] = -1
            except Exception:
                output_positions[i] = -1

        return output_positions

    def infer_sequence_to_labels_probs(self, sequences, conditional_input=None):
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self, variant_sequences, ref_sequences, conditional_input=None
    ):
        return None

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.mlm_head_loaded:
            return None, None

        batch_size = len(sequences)

        masked_sequences = []
        for seq in sequences:
            masked_seq = seq[:variant_pos] + self.tokenizer.mask_token + seq[variant_pos + 1 :]
            masked_sequences.append(masked_seq)

        encoded = self.tokenizer(
            masked_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: val.to(self.device) for key, val in encoded.items()}

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (encoded["input_ids"] == mask_token_id).nonzero(as_tuple=True)

        if len(mask_positions[0]) != batch_size:
            return None, None

        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)

        nucleotide_token_ids = {}
        for nuc in ["A", "T", "C", "G"]:
            tokens = self.tokenizer.encode(nuc, add_special_tokens=False)
            if len(tokens) == 1:
                nucleotide_token_ids[nuc] = tokens[0]
            else:
                tokens = self.tokenizer.encode(nuc.lower(), add_special_tokens=False)
                if len(tokens) == 1:
                    nucleotide_token_ids[nuc] = tokens[0]

        if len(nucleotide_token_ids) < 4:
            return None, None

        variant_probs_list = []
        reference_probs_list = []

        for i in range(batch_size):
            batch_mask_indices = (mask_positions[0] == i).nonzero(as_tuple=True)[0]
            if len(batch_mask_indices) == 0:
                variant_probs_list.append(0.0)
                reference_probs_list.append(0.0)
                continue

            mask_pos = mask_positions[1][batch_mask_indices[0]]
            var_nuc = variant_letters[i].upper()
            ref_nuc = reference_letters[i].upper()
            var_token_id = nucleotide_token_ids.get(var_nuc)
            ref_token_id = nucleotide_token_ids.get(ref_nuc)

            if var_token_id is None or ref_token_id is None:
                variant_probs_list.append(0.0)
                reference_probs_list.append(0.0)
                continue

            variant_probs_list.append(probs[i, mask_pos, var_token_id].item())
            reference_probs_list.append(probs[i, mask_pos, ref_token_id].item())

        return (
            np.array(variant_probs_list, dtype=np.float32),
            np.array(reference_probs_list, dtype=np.float32),
        )

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"], strict=False)
            self.mlm_head_loaded = True
            print("Loaded model weights (including MLM head)")
        else:
            self.model.load_state_dict(state, strict=False)
            self.mlm_head_loaded = True
            print("Loaded model weights (direct state_dict)")

    def save_checkpoint(self, checkpoint_path: str, extra_state: dict = None):
        import os

        os.makedirs(
            os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".",
            exist_ok=True,
        )

        checkpoint = {"model_state_dict": self.model.state_dict()}
        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    def get_tokenizer(self):
        return self.tokenizer

    def load_pretrained_embeddings(self, model_name: str = None):
        model_name = model_name or self.model_name
        print(f"Loading pretrained token embeddings from: {model_name}")

        pretrained = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.bert.embeddings.word_embeddings.load_state_dict(
            pretrained.bert.embeddings.word_embeddings.state_dict()
        )
        del pretrained
        print("  -> Pretrained token embeddings loaded successfully")

    def freeze_token_embeddings(self):
        for param in self.model.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = False
        print("  -> Token embeddings frozen (requires_grad=False)")

    def all_parameters(self):
        return self.model.parameters()
