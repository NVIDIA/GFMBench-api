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
"""Token -> fixed-bin pooling for binned regression, built on public model methods only."""
from typing import Any, List

import numpy as np
import torch


def get_valid_token_counts(model: Any, sequences: List[str], num_tokens: int) -> torch.Tensor:
    """Return the number of valid (non-padding) output positions of each sequence.

    Derived from ``model.sequence_pos_to_prob_pos`` applied to each sequence's last
    nucleotide: the output position of the last nucleotide, plus one. This is exact
    when the last nucleotide maps to the last output position (single-nucleotide,
    BPE and uniform k-mer tokenizers); a position that cannot be determined
    (``-1``) raises ``ValueError``.

    Args:
        model: model implementing ``sequence_pos_to_prob_pos``.
        sequences: batch of DNA sequences.
        num_tokens: number of output positions ``T`` returned for the batch.

    Returns:
        ``[batch_size]`` long tensor with values in ``[1, num_tokens]``.
    """
    counts = torch.full((len(sequences),), num_tokens, dtype=torch.long)
    by_length = {}
    for i, sequence in enumerate(sequences):
        if len(sequence) > 0:
            by_length.setdefault(len(sequence), []).append(i)
    for length, indices in by_length.items():
        positions = model.sequence_pos_to_prob_pos([sequences[i] for i in indices], length - 1)
        positions = torch.as_tensor(positions, dtype=torch.long)
        valid = positions >= 0
        if not bool(valid.all()):
            raise ValueError(
                "sequence_pos_to_prob_pos could not map the last nucleotide to an output "
                "position (returned -1); cannot determine the valid token count"
            )
        counts[torch.tensor(indices)] = positions + 1
    return counts.clamp(min=1, max=num_tokens)


def to_float_tensor(values: Any, device: Any) -> torch.Tensor:
    """Convert a tensor or array-like to a detached float32 tensor on ``device``."""
    if torch.is_tensor(values):
        return values.detach().to(device).float()
    return torch.from_numpy(np.asarray(values)).to(device).float()


def pool_tokens_to_bins(
    token_values: torch.Tensor, valid_counts: torch.Tensor, num_bins: int
) -> torch.Tensor:
    """Mean-pool per-token values into ``num_bins`` bins by fractional token position.

    Each valid token is assigned wholly to bin ``floor(index * num_bins / n_valid)``
    where ``n_valid`` is the sample's valid-token count, then values are averaged
    within each bin. Tokenizer-agnostic; tokens past ``valid_counts`` are ignored.
    When a sample has fewer valid tokens than bins, each empty bin takes the value of
    the token covering its start.

    Args:
        token_values: ``[B, T, D]`` per-token values (embeddings or predictions).
        valid_counts: ``[B]`` number of valid leading tokens per sample.
        num_bins: number of output bins.

    Returns:
        ``[B, num_bins, D]`` per-bin means.
    """
    B, T, D = token_values.shape
    device, dtype = token_values.device, token_values.dtype
    index = torch.arange(T, device=device)[None, :]  # [1, T] long
    counts = valid_counts.to(device=device, dtype=torch.long)[:, None].clamp(min=1)  # [B, 1]
    mask = (index < counts).to(dtype)  # [B, T]

    # Integer arithmetic keeps exact bin boundaries exact.
    bin_idx = ((index * num_bins) // counts).clamp(max=num_bins - 1) * mask.long()

    summed = torch.zeros(B, num_bins, D, device=device, dtype=dtype)
    summed.scatter_add_(1, bin_idx.unsqueeze(-1).expand(B, T, D), token_values * mask.unsqueeze(-1))
    bin_counts = torch.zeros(B, num_bins, device=device, dtype=dtype)
    bin_counts.scatter_add_(1, bin_idx, mask)
    pooled = summed / bin_counts.clamp(min=1.0).unsqueeze(-1)

    # Bins that received no token (fewer valid tokens than bins) take the value of the
    # token covering the bin's start, so they are not scored as zeros.
    if bool((bin_counts > 0).all()):
        return pooled
    bins = torch.arange(num_bins, device=device)[None, :]  # [1, num_bins]
    source = ((bins * counts) // num_bins).clamp(max=T - 1)  # [B, num_bins]
    fallback = torch.gather(token_values, 1, source.unsqueeze(-1).expand(B, num_bins, D))
    return torch.where((bin_counts > 0).unsqueeze(-1), pooled, fallback)
