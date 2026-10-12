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

"""Smoke-test fixtures and MockGFMModel."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "data"
SMOKE_TASK_CONFIG = {
    "max_num_samples": 8,
    "batch_size": 4,
    "max_sequence_length": 64,
}


@pytest.fixture
def fixtures_root(tmp_path) -> Path:
    """Copy packaged fixture data into an isolated temp directory per test."""
    import shutil

    dest = tmp_path / "data"
    shutil.copytree(FIXTURES_DIR, dest)
    return dest


@pytest.fixture
def smoke_task_config() -> dict:
    return dict(SMOKE_TASK_CONFIG)


class MockGFMModel:
    """Deterministic stand-in for a GFM that implements all inference methods."""

    def __init__(
        self,
        num_labels: int = 2,
        hidden_dim: int = 16,
        seed: int = 42,
    ) -> None:
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self._rng = np.random.RandomState(seed)

    def eval(self) -> None:
        pass

    def infer_sequence_to_labels_probs(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        batch_size = len(sequences)
        probs = np.zeros((batch_size, self.num_labels), dtype=np.float64)
        for i in range(batch_size):
            probs[i, i % self.num_labels] = 0.75
            probs[i, (i + 1) % self.num_labels] = 0.25
        return probs / probs.sum(axis=1, keepdims=True)

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.infer_sequence_to_labels_probs(variant_sequences, conditional_input)

    def infer_sequence_to_sequence(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(sequences)
        seq_len = max(len(s) for s in sequences)
        seq_probs = np.full(
            (batch_size, seq_len),
            0.25,
            dtype=np.float64,
        )
        embeddings = self._rng.randn(batch_size, seq_len, self.hidden_dim)
        reprs = self._rng.randn(batch_size, self.hidden_dim)
        # Lower similarity for odd-indexed samples (pairs with label=1 in fixtures).
        reprs[1::2] += 2.0
        return seq_probs, embeddings, reprs

    def sequence_pos_to_prob_pos(
        self,
        sequences: List[str],
        pos: int,
        conditional_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.full(len(sequences), pos, dtype=np.int32)

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = len(sequences)
        variant_probs = np.full(batch_size, 0.7, dtype=np.float64)
        reference_probs = np.full(batch_size, 0.6, dtype=np.float64)
        variant_probs[1::2] = 0.3
        return variant_probs, reference_probs


@pytest.fixture
def mock_model() -> MockGFMModel:
    return MockGFMModel()
