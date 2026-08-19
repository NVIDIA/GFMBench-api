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
# - https://huggingface.co/LongSafari/hyenadna-tiny-16k-seqlen-d128-hf — Apache-2.0

"""HyenaDNA causal LM adapter for GFMBench (embeddings + next-token probs)."""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class HyenaDNAModel(nn.Module):
    """LongSafari HyenaDNA HF causal LM for GFMBench zero-shot / probing / FT.

    Pretrained with causal next-token prediction, so masked-token scoring paths
    are unsupported and variant effects are scored from next-token probabilities.
    """

    HUGGINGFACE_MODEL_NAME = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"

    def __init__(
        self,
        device: str = "cpu",
        model_name: Optional[str] = None,
        max_length: int = 8192,
        pretrained: bool = True,
    ):
        super().__init__()
        self.device = device
        self.max_length = int(max_length)
        self.pretrained = pretrained
        # Prefer explicit arg, then HYENADNA_MODEL_NAME (local snapshot), then HF id.
        self.model_name = (
            model_name
            or os.environ.get("HYENADNA_MODEL_NAME")
            or self.HUGGINGFACE_MODEL_NAME
        )

        print(f"Loading HyenaDNA model: {self.model_name} (pretrained={pretrained})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        load_kwargs = {"trust_remote_code": True}
        if device == "cpu":
            load_kwargs["attn_implementation"] = "eager"

        if pretrained:
            hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            hf_model = AutoModelForCausalLM.from_config(config, **load_kwargs)
            print("  -> HyenaDNA initialized from config (random weights)")

        self.model = hf_model
        self.add_module("model", hf_model)
        self.model.to(device)

        self.hidden_dim = int(self.model.config.d_model)
        self._pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0

        print(
            f"HyenaDNA loaded. Hidden dim: {self.hidden_dim}, "
            f"max_length: {self.max_length}"
        )

    def _ensure_attention_mask(self, encoded: dict) -> dict:
        if "attention_mask" in encoded and encoded["attention_mask"] is not None:
            return encoded
        input_ids = encoded["input_ids"]
        mask = (input_ids != self._pad_id).to(input_ids.device, dtype=torch.long)
        out = dict(encoded)
        out["attention_mask"] = mask
        return out

    @staticmethod
    def _last_valid_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is None or attention_mask.numel() == 0:
            raise ValueError("attention_mask required")
        t = attention_mask.shape[1]
        flipped = torch.flip(attention_mask.to(torch.long), dims=[1])
        last_rel = flipped.argmax(dim=1)
        return (t - 1 - last_rel).to(torch.long)

    def state_dict(self, prefix: str = "", keep_vars: bool = False):
        return self.model.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in self.model.named_parameters(prefix="", recurse=recurse):
            yield name, param

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = attention_mask
        out = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1], out.logits

    def embeddings_to_representative(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                embeddings.shape[:2], device=embeddings.device, dtype=torch.long
            )
        b = embeddings.shape[0]
        last_idx = self._last_valid_token_index(attention_mask)
        h = embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return h[torch.arange(b, device=embeddings.device), last_idx, :]

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_tokenizer(self):
        return self.tokenizer

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device if isinstance(device, str) else str(device)
        return self

    def tokenize_sequence(self, sequence: str) -> List[int]:
        return self.tokenizer.encode((sequence or "").upper(), add_special_tokens=True)

    def tokenize(self, sequences: List[str]) -> dict:
        encoded = self.tokenizer(
            [(s or "").upper() for s in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return self._ensure_attention_mask(encoded)

    def _representative_from_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        b = hidden.shape[0]
        last_idx = self._last_valid_token_index(attention_mask)
        h = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return h[torch.arange(b, device=hidden.device), last_idx, :]

    def _sequence_to_representative(
        self, sequences: List[str], conditional_input=None
    ) -> torch.Tensor:
        """Representative embeddings as torch tensors for supervised heads."""
        del conditional_input
        encoded = self.tokenize(sequences)
        out = self.model(
            input_ids=encoded["input_ids"],
            output_hidden_states=True,
            return_dict=True,
        )
        return self.embeddings_to_representative(out.hidden_states[-1], encoded["attention_mask"])

    def infer_sequence_to_sequence_representative(
        self,
        sequences: List[str],
        conditional_input=None,
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del conditional_input
        encoded = self.tokenize(sequences)
        if require_grad:
            out = self.model(
                input_ids=encoded["input_ids"],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                out = self.model(
                    input_ids=encoded["input_ids"],
                    output_hidden_states=True,
                    return_dict=True,
                )
        sequence_embeddings = out.hidden_states[-1]
        rep = self._representative_from_hidden(sequence_embeddings, encoded["attention_mask"])
        return sequence_embeddings, rep

    def _next_token_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Causal next-token probabilities aligned with ``input_ids``.

        ``logits[:, t]`` predicts token ``t + 1``. For ``t >= 1``, store
        ``P(input_ids[t] | prefix)`` from ``logits[:, t - 1]``. Position 0 has
        no causal target; use uniform ``1 / V`` (Evo2 convention). Pads are
        zeroed via ``attention_mask``.
        """
        vocab = logits.shape[-1]
        log_p = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.full(
            input_ids.shape,
            -float(np.log(vocab)),
            device=logits.device,
            dtype=logits.dtype,
        )
        if input_ids.shape[1] > 1:
            token_log_probs[:, 1:] = (
                log_p[:, :-1, :]
                .gather(2, input_ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )
        return token_log_probs.exp().clamp(0.0, 1.0) * attention_mask.to(logits.dtype)

    def infer_sequence_to_sequence(
        self, sequences: List[str], conditional_input=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(sequence_probs, embeddings, representative)``.

        ``sequence_probs[b, t]`` is the probability of the token at index ``t``
        under next-token prediction (see ``BaseGFMModel`` / Evo2 AR scoring).
        """
        del conditional_input
        encoded = self.tokenize(sequences)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        batch_size = hidden_states.shape[0]
        last_idx = self._last_valid_token_index(attention_mask)
        representative = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), last_idx, :
        ]
        sequence_probs = self._next_token_probs(logits, input_ids, attention_mask)
        return (
            sequence_probs.detach().cpu().numpy().astype(np.float32, copy=False),
            hidden_states.detach().cpu().numpy(),
            representative.detach().cpu().numpy(),
        )

    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        """Map DNA index ``pos`` to the token index where ``P(token)`` is stored.

        Probs are aligned with ``input_ids`` (P of token ``t`` at index ``t``),
        so this returns the absolute token index of that nucleotide (not ``pos-1``).
        """
        encoded = self.tokenize(sequences)
        attn = encoded["attention_mask"]
        bsz = attn.shape[0]
        out = np.full(bsz, -1, dtype=np.int32)
        for b in range(bsz):
            seq = (sequences[b] or "").upper()
            if pos < 0 or pos >= len(seq):
                continue
            row = attn[b]
            idxs = (row == 1).nonzero(as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue
            start = int(idxs[0].item())
            tp = start + int(pos)
            if 0 <= tp < attn.shape[1] and row[tp].item() == 1:
                out[b] = tp
        return out

    def infer_sequence_to_labels_probs(
        self, sequences: List[str], conditional_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        return None

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Causal LM: masked MLM metrics are not supported (use sum_probs_llr instead).
        return None, None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"], strict=False)
        else:
            self.load_state_dict(state, strict=False)
        print("Checkpoint loaded successfully")
