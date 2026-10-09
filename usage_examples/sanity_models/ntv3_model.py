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

# Third-party URL notices for this file (see THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/InstaDeepAI/NTv3_8M_pre — LicenseRef-InstaDeep-NTv3-NonCommercial
# - https://huggingface.co/InstaDeepAI/NTv3_100M_pre — LicenseRef-InstaDeep-NTv3-NonCommercial
"""GFM-Bench adapter for InstaDeep Nucleotide Transformer v3 (NTv3).

Requires Transformers and Hugging Face access to the InstaDeepAI NTv3
checkpoints (``trust_remote_code=True``).

Supported ``model_name`` shortcuts:
    - ``NTv3_8M_pre``  -> ``InstaDeepAI/NTv3_8M_pre``
    - ``NTv3_100M_pre`` -> ``InstaDeepAI/NTv3_100M_pre``
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel

logger = logging.getLogger(__name__)

NTV3_SEQ_MULTIPLE = 128


def resolve_hf_model_id(model_name: Optional[str]) -> str:
    if model_name is None:
        model_name = "NTv3_8M_pre"
    if model_name == "NTv3_8M_pre":
        return "InstaDeepAI/NTv3_8M_pre"
    if model_name == "NTv3_100M_pre":
        return "InstaDeepAI/NTv3_100M_pre"
    if model_name.startswith("InstaDeepAI/"):
        return model_name
    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        "Expected 'NTv3_8M_pre', 'NTv3_100M_pre', or a HuggingFace model id."
    )


def build_ntv3_mlm(
    model_name: str,
    *,
    pretrained: bool = True,
    local_files_only: bool = False,
) -> Tuple[AutoTokenizer, AutoModelForMaskedLM]:
    hf_kwargs = {"trust_remote_code": True, "local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
    hf_config = AutoConfig.from_pretrained(model_name, **hf_kwargs)
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(model_name, config=hf_config, **hf_kwargs)
    else:
        model = AutoModelForMaskedLM.from_config(hf_config, trust_remote_code=True)
    return tokenizer, model


def get_ntv3_hidden_dim(model: AutoModelForMaskedLM) -> int:
    return int(model.core.config.embed_dim)


def _get_pad_token_id(tokenizer: AutoTokenizer) -> int:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    return int(pad_token_id)


def _pad_to_seq_multiple(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    multiple: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = input_ids.shape[1]
    remainder = seq_len % multiple
    if remainder == 0:
        return input_ids, attention_mask

    pad_len = multiple - remainder
    pad_ids = torch.full(
        (input_ids.shape[0], pad_len),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    pad_mask = torch.zeros(
        (attention_mask.shape[0], pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return (
        torch.cat([input_ids, pad_ids], dim=1),
        torch.cat([attention_mask, pad_mask], dim=1),
    )


def tokenize_sequences(
    tokenizer: AutoTokenizer,
    sequences: List[str],
    *,
    device: Union[str, torch.device],
    max_length: int,
    pad_to_multiple: int = NTV3_SEQ_MULTIPLE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=pad_to_multiple,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return _pad_to_seq_multiple(
        input_ids,
        attention_mask,
        multiple=pad_to_multiple,
        pad_token_id=_get_pad_token_id(tokenizer),
    )


class NucleotideTransformerV3Model(BaseGFMModel):
    """InstaDeep NTv3 (8M / 100M) for GFM-Bench zero-shot and supervised evaluation."""

    def __init__(
        self,
        device: str = "cuda",
        max_length: int = 8192,
        pretrained: bool = True,
        model_name: Optional[str] = None,
        sequence_representative_type: str = "mean",
        local_files_only: bool = False,
        use_autocast: bool = False,
    ) -> None:
        self.device = device
        self.model_name = resolve_hf_model_id(model_name)
        self.max_length = max_length
        self.pretrained = pretrained
        self.sequence_representative_type = sequence_representative_type
        self.cls_index = 0
        self.mlm_head_loaded = True
        self._use_autocast = bool(use_autocast) and isinstance(device, str) and device.startswith("cuda")
        self._autocast_dtype = torch.bfloat16

        logger.info(
            "Loading NucleotideTransformerV3: %s (pretrained=%s)",
            self.model_name,
            pretrained,
        )
        self.tokenizer, mlm_full = build_ntv3_mlm(
            self.model_name,
            pretrained=pretrained,
            local_files_only=local_files_only,
        )
        if not pretrained:
            logger.info("Initialized NTv3 with random weights")

        self.model = mlm_full
        self.model.to(self.device)
        self.model.eval()
        self.hidden_dim = get_ntv3_hidden_dim(mlm_full)
        logger.info("NTv3 loaded. hidden_dim=%s, max_length=%s", self.hidden_dim, self.max_length)

    def _add_cls_token_if_needed(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sequence_representative_type != "CLS":
            return input_ids, attention_mask
        cls_token_id = self.tokenizer.cls_token_id
        if cls_token_id is None:
            raise ValueError("CLS token not found in NTv3 tokenizer")
        batch_size = input_ids.shape[0]
        device = input_ids.device
        cls_tokens = torch.full((batch_size, 1), cls_token_id, dtype=input_ids.dtype, device=device)
        input_ids_with_cls = torch.cat([cls_tokens, input_ids[:, :-1]], dim=1)
        cls_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
        attention_mask_with_cls = torch.cat([cls_mask, attention_mask[:, :-1]], dim=1)
        return input_ids_with_cls, attention_mask_with_cls

    def _encode(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = tokenize_sequences(
            self.tokenizer,
            sequences,
            device=self.device,
            max_length=self.max_length,
            pad_to_multiple=NTV3_SEQ_MULTIPLE,
        )
        return self._add_cls_token_if_needed(input_ids, attention_mask)

    def embeddings_to_representative(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.sequence_representative_type == "CLS":
            return embeddings[:, self.cls_index, :]
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.float)
        else:
            attention_mask = attention_mask.float()
        expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        summed = torch.sum(embeddings * expanded, dim=1)
        denom = torch.clamp(expanded.sum(1), min=1e-9)
        return summed / denom

    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        if self.sequence_representative_type == "CLS":
            return np.full(len(sequences), pos + 1, dtype=np.int32)
        return np.full(len(sequences), pos, dtype=np.int32)

    def infer_sequence_to_sequence(
        self, sequences: List[str], conditional_input: Any = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        input_ids, attention_mask = self._encode(sequences)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=self._autocast_dtype, enabled=self._use_autocast
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"][-1]
            representative = self.embeddings_to_representative(hidden_states, attention_mask)
        return None, hidden_states.float().cpu().numpy(), representative.float().cpu().numpy()

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input: Any = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.mlm_head_loaded:
            return None, None

        batch_size = len(sequences)
        input_ids, attention_mask = self._encode(sequences)
        mask_token_id = self.tokenizer.mask_token_id
        shifted_pos = variant_pos + 1 if self.sequence_representative_type == "CLS" else variant_pos

        masked_input_ids = input_ids.clone()
        valid_indices: list[int] = []
        for i in range(batch_size):
            if 0 <= variant_pos < len(sequences[i]) and shifted_pos < input_ids.shape[1]:
                masked_input_ids[i, shifted_pos] = mask_token_id
                valid_indices.append(i)

        if not valid_indices:
            return np.zeros(batch_size), np.zeros(batch_size)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=self._autocast_dtype, enabled=self._use_autocast
        ):
            outputs = self.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        probs = torch.softmax(outputs["logits"].float(), dim=-1)

        variant_probs: list[float] = []
        reference_probs: list[float] = []
        for i in range(batch_size):
            if i not in valid_indices:
                variant_probs.append(0.0)
                reference_probs.append(0.0)
                continue
            ref_id = self.tokenizer.convert_tokens_to_ids(reference_letters[i])
            var_id = self.tokenizer.convert_tokens_to_ids(variant_letters[i])
            if ref_id == self.tokenizer.unk_token_id or var_id == self.tokenizer.unk_token_id:
                variant_probs.append(0.0)
                reference_probs.append(0.0)
            else:
                variant_probs.append(probs[i, shifted_pos, var_id].item())
                reference_probs.append(probs[i, shifted_pos, ref_id].item())

        return np.array(variant_probs, dtype=np.float32), np.array(reference_probs, dtype=np.float32)

    def _sequence_to_representative(
        self, sequences: List[str], conditional_input: Any = None
    ) -> torch.Tensor:
        input_ids, attention_mask = self._encode(sequences)
        with torch.autocast(
            device_type="cuda", dtype=self._autocast_dtype, enabled=self._use_autocast
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"][-1]
            representative = self.embeddings_to_representative(hidden_states, attention_mask)
        return representative.float()

    def infer_sequence_to_labels_probs(
        self, sequences: List[str], conditional_input: Any = None
    ) -> Optional[np.ndarray]:
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Any = None,
    ) -> Optional[np.ndarray]:
        return None

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def eval(self) -> "NucleotideTransformerV3Model":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "NucleotideTransformerV3Model":
        self.model.train(mode)
        return self

    def to(self, device: str) -> "NucleotideTransformerV3Model":
        self.model.to(device)
        self.device = device
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def load_checkpoint(self, checkpoint_path: str) -> "NucleotideTransformerV3Model":
        logger.info("Loading NTv3 checkpoint: %s", checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.mlm_head_loaded = True
        logger.info("NTv3 checkpoint loaded")
        return self
