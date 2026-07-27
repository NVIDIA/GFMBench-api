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

"""
Unit tests for SequenceInferenceCache.

All tests use lightweight mock functions and plain numpy arrays — no model or GPU
required. Each test creates its own fresh cache instance.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from gfmbench_api.utils.caching_utils import SequenceInferenceCache


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_fn(seq_outputs: dict) -> Mock:
    """Return a Mock whose side_effect stacks per-sequence arrays into a batch."""
    return Mock(side_effect=lambda seqs, *extra: np.stack([seq_outputs[s] for s in seqs]))


# ---------------------------------------------------------------------------
# Group 1 — Core cache semantics
# ---------------------------------------------------------------------------

def test_full_hit_skips_fn():
    """Second identical call must not invoke fn and must return the same values."""
    fn = _make_fn({"seq_A": np.array([1.0, 2.0, 3.0])})
    cache = SequenceInferenceCache()

    result1 = cache.cached_call(fn, ["seq_A"])
    result2 = cache.cached_call(fn, ["seq_A"])

    assert fn.call_count == 1
    np.testing.assert_array_equal(result1, result2)


def test_partial_hit_calls_fn_for_misses_only():
    """fn must be called only for missing sequences; merged result must be correct."""
    outputs = {
        "seq_A": np.array([1.0, 0.0, 0.0]),
        "seq_B": np.array([0.0, 1.0, 0.0]),
        "seq_C": np.array([0.0, 0.0, 1.0]),
    }
    fn = _make_fn(outputs)
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"])
    assert fn.call_count == 1

    result = cache.cached_call(fn, ["seq_B", "seq_C", "seq_A"])

    assert fn.call_count == 2
    # Verify fn was called with only the two missed sequences
    assert fn.call_args_list[-1][0][0] == ["seq_B", "seq_C"]
    # Verify every position in the merged result is correct
    np.testing.assert_array_almost_equal(result[0], outputs["seq_B"])
    np.testing.assert_array_almost_equal(result[1], outputs["seq_C"])
    np.testing.assert_array_almost_equal(result[2], outputs["seq_A"])


def test_duplicate_sequences_deduplicated():
    """Repeated sequence in a batch must cause only one fn call; result broadcast."""
    output = np.array([1.0, 2.0, 3.0])
    fn = _make_fn({"seq_A": output})
    cache = SequenceInferenceCache()

    result = cache.cached_call(fn, ["seq_A", "seq_A"])

    assert fn.call_count == 1
    assert result.shape[0] == 2
    np.testing.assert_array_almost_equal(result[0], output)
    np.testing.assert_array_almost_equal(result[1], output)


# ---------------------------------------------------------------------------
# Group 2 — disable and clear
# ---------------------------------------------------------------------------

def test_disable_bypasses_cache():
    """disable=True must bypass both read and write; cache stays empty."""
    fn = _make_fn({"seq_A": np.array([1.0, 2.0])})
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"], disable=True)
    cache.cached_call(fn, ["seq_A"], disable=True)
    # Cache is still empty, so this normal call is also a miss
    cache.cached_call(fn, ["seq_A"])

    assert fn.call_count == 3


def test_clear_invalidates_entries():
    """After clear(), previously cached sequences must be treated as misses."""
    fn = _make_fn({"seq_A": np.array([1.0, 2.0])})
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"])  # populate
    assert cache.current_size_bytes > 0
    cache.clear()
    assert cache.current_size_bytes == 0
    cache.cached_call(fn, ["seq_A"])  # must be a miss again

    assert fn.call_count == 2


def test_cache_size_zero_disables_caching():
    """max_size_gb=0 must bypass both read and write; cache stays empty."""
    fn = _make_fn({"seq_A": np.array([1.0, 2.0])})
    cache = SequenceInferenceCache(max_size_gb=0)

    cache.cached_call(fn, ["seq_A"])
    cache.cached_call(fn, ["seq_A"])
    cache.cached_call(fn, ["seq_A"], disable=False)

    assert fn.call_count == 3
    assert cache.current_size_bytes == 0
    assert len(cache._cache) == 0


def test_cache_size_limit_stops_writes_but_keeps_hits():
    """When full, misses are computed but not stored; prior entries remain readable."""
    output_a = np.ones(512, dtype=np.float64)
    output_b = np.ones(512, dtype=np.float64) * 2
    fn = _make_fn({"seq_A": output_a, "seq_B": output_b})
    # 4 KiB cap: first entry fits, second does not
    cache = SequenceInferenceCache(max_size_gb=4096 / 1024**3)

    cache.cached_call(fn, ["seq_A"])
    assert fn.call_count == 1
    assert cache.current_size_bytes > 0
    first_size = cache.current_size_bytes

    cache.cached_call(fn, ["seq_B"])
    assert fn.call_count == 2
    assert cache.current_size_bytes == first_size
    assert cache.is_full

    cache.cached_call(fn, ["seq_A"])
    assert fn.call_count == 2
    result = cache.cached_call(fn, ["seq_A"])
    assert fn.call_count == 2
    np.testing.assert_array_equal(result[0], output_a)


# ---------------------------------------------------------------------------
# Group 3 — Output type fidelity
# ---------------------------------------------------------------------------

def test_torch_tensor_restored_to_correct_dtype_and_device():
    """Torch tensor output must survive a cache round-trip with the original dtype/device."""
    torch = pytest.importorskip("torch")

    original = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    fn = Mock(side_effect=lambda seqs, *extra: original)
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"])       # cold — populate
    result = cache.cached_call(fn, ["seq_A"])  # warm — full hit

    assert fn.call_count == 1
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.device == original.device


def test_tuple_output_all_slots_restored():
    """Tuple return (ndarray, ndarray, None) must have all slots correct on a cache hit."""
    probs = np.array([[0.8, 0.2]])      # shape (1, 2)
    embs = np.array([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
    fn = Mock(side_effect=lambda seqs, *extra: (probs, embs, None))
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"])       # cold — populate
    result = cache.cached_call(fn, ["seq_A"])  # warm — full hit

    assert fn.call_count == 1
    out_probs, out_embs, out_none = result
    assert out_none is None
    np.testing.assert_array_almost_equal(out_probs, probs)
    np.testing.assert_array_almost_equal(out_embs, embs)


# ---------------------------------------------------------------------------
# Group 4 — Key and merge correctness
# ---------------------------------------------------------------------------

def test_extra_args_change_cache_key():
    """Same sequence with different scalar extra arg must produce two separate cache entries."""
    output = np.array([1.0, 2.0])
    fn = Mock(side_effect=lambda seqs, extra: np.stack([output for _ in seqs]))
    cache = SequenceInferenceCache()

    cache.cached_call(fn, ["seq_A"], 0)
    cache.cached_call(fn, ["seq_A"], 1)

    assert fn.call_count == 2


def test_variable_length_embeddings_padded():
    """2-D per-sequence rows of different lengths must be zero-padded on merge."""
    D = 4
    short_row = np.ones((3, D), dtype=np.float32)      # seq_A: length 3
    long_row = np.ones((5, D), dtype=np.float32) * 2   # seq_B: length 5

    def _embed(seqs):
        rows = [short_row if s == "seq_A" else long_row for s in seqs]
        max_l = max(r.shape[0] for r in rows)
        out = np.zeros((len(seqs), max_l, D), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
        return out

    fn = Mock(side_effect=_embed)
    cache = SequenceInferenceCache()

    # Populate each sequence individually (stored as 2-D rows)
    cache.cached_call(fn, ["seq_A"])
    cache.cached_call(fn, ["seq_B"])
    assert fn.call_count == 2

    # Full-hit retrieval triggers _pad_embedding_rows
    result = cache.cached_call(fn, ["seq_A", "seq_B"])

    assert fn.call_count == 2  # no new fn call
    assert result.shape == (2, 5, D)
    np.testing.assert_array_equal(result[0, :3], short_row)
    np.testing.assert_array_equal(result[0, 3:], np.zeros((2, D)))
    np.testing.assert_array_equal(result[1, :5], long_row)
