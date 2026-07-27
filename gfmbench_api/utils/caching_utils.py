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

from typing import Any, Callable, Hashable, List, Optional, Sequence, Tuple

import logging
import numpy as np
import torch

# OutputSpec tags cached return types so hits can be restored to the caller format.
# ("none",) | ("numpy", dtype) | ("torch", device, dtype) | ("list",) | ("raw",)
OutputSpec = Tuple[Any, ...]

# =============================================================================
# Sequence Inference Cache
# =============================================================================


class SequenceInferenceCache:
    """
    Per-sequence cache for batch inference calls.

    Wraps a callable whose first positional argument is a list of sequence strings.
    Cache keys are derived from each sequence plus any batch-shared extra arguments.
    On partial cache hits, only missed sequences are forwarded to the wrapped function
    in their original relative order; outputs are merged back to the full batch order.

    Cached values are always stored on CPU (as numpy arrays). On return, outputs are
    restored to the same type/device as the wrapped function would produce (e.g. GPU
    tensors for training forwards, numpy for inference APIs).

  Optional ``max_size_gb`` caps total cached value bytes. When the cap is reached,
  cache hits are still served but new entries are not stored until ``clear()``.
  ``max_size_gb=0`` disables caching entirely (no reads or writes).
    """

    def __init__(self, max_size_gb: Optional[float] = None) -> None:
        self._cache: dict = {}
        self._output_specs: Optional[Tuple[OutputSpec, ...]] = None
        self._current_size_bytes: int = 0
        if max_size_gb is None:
            self._max_size_bytes: Optional[int] = None
        elif max_size_gb == 0:
            self._max_size_bytes = 0
        else:
            self._max_size_bytes = int(max_size_gb * 1024**3)
        self._full_warning_logged = False

    @property
    def current_size_bytes(self) -> int:
        """Total nbytes of cached value arrays currently stored."""
        return self._current_size_bytes

    @property
    def max_size_bytes(self) -> Optional[int]:
        """Configured byte cap, or None when unlimited."""
        return self._max_size_bytes

    @property
    def is_full(self) -> bool:
        """True when a positive byte cap is set and no further entries can fit."""
        if self._max_size_bytes is None or self._max_size_bytes == 0:
            return False
        return self._current_size_bytes >= self._max_size_bytes

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._output_specs = None
        self._current_size_bytes = 0
        self._full_warning_logged = False

    def cached_call(self, fn: Callable[..., Any], *args: Any, disable: bool = False) -> Any:
        """
        Call ``fn`` with per-sequence caching on the first positional argument.

        Args:
            fn: Callable whose first arg is ``List[str]`` (the batch of sequences).
            *args: Positional arguments passed to ``fn``; ``args[0]`` is the sequence batch.
            disable: If True, call ``fn`` directly without reading or writing the cache.

        Returns:
            Same value structure as ``fn`` would return for the full batch.
        """
        if disable or self._max_size_bytes == 0:
            if not args:
                return fn(*args)
            sequences = args[0]
            if isinstance(sequences, tuple):
                args = (list(sequences), *args[1:])
            return fn(*args)

        if not args:
            return fn(*args)

        sequences = args[0]
        if isinstance(sequences, tuple):
            sequences = list(sequences)
            args = (sequences, *args[1:])
        if not isinstance(sequences, list) or len(sequences) == 0:
            return fn(*args)

        extra_args = args[1:]
        batch_size = len(sequences)
        keys = [self._make_key(sequences[i], extra_args) for i in range(batch_size)]

        per_index_rows: List[Optional[Tuple[Any, ...]]] = [None] * batch_size
        miss_key_to_indices: dict = {}

        for i, key in enumerate(keys):
            if key in self._cache:
                per_index_rows[i] = self._cache[key]
            else:
                miss_key_to_indices.setdefault(key, []).append(i)

        if not miss_key_to_indices:
            return self._assemble_and_restore(per_index_rows)

        unique_miss_keys = sorted(
            miss_key_to_indices.keys(),
            key=lambda key: miss_key_to_indices[key][0],
        )
        representative_indices = [miss_key_to_indices[key][0] for key in unique_miss_keys]
        filtered_args = self._filter_args(sequences, extra_args, representative_indices, batch_size)
        fresh_result = fn(*filtered_args)
        fresh_outputs = fresh_result if isinstance(fresh_result, tuple) else (fresh_result,)

        if self._output_specs is None:
            self._output_specs = tuple(self._get_output_spec(out) for out in fresh_outputs)

        for uk_idx, key in enumerate(unique_miss_keys):
            row_outputs = tuple(
                self._to_cache_value(self._slice_row(out, uk_idx) if out is not None else None)
                for out in fresh_outputs
            )
            entry_size = self._entry_size_bytes(row_outputs)
            if self._can_store(entry_size):
                was_not_full = not self.is_full
                self._cache[key] = row_outputs
                self._current_size_bytes += entry_size
                if was_not_full and self.is_full:
                    self._warn_cache_full_once()
            elif self.is_full:
                self._warn_cache_full_once()
            for i in miss_key_to_indices[key]:
                per_index_rows[i] = row_outputs

        if len(unique_miss_keys) == batch_size:
            return fresh_result

        return self._assemble_and_restore(per_index_rows)

    @staticmethod
    def _entry_size_bytes(row: Tuple[Any, ...]) -> int:
        """Sum nbytes of numpy arrays in a cached row."""
        total = 0
        for value in row:
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return total

    def _can_store(self, entry_size: int) -> bool:
        """Return True if a row of ``entry_size`` bytes may be written to the cache."""
        if self._max_size_bytes is None:
            return True
        if self._max_size_bytes == 0:
            return False
        return self._current_size_bytes + entry_size <= self._max_size_bytes

    def _warn_cache_full_once(self) -> None:
        """Log a one-time warning when the cache has reached its byte cap."""
        if self._full_warning_logged:
            return
        if self._max_size_bytes is None or self._max_size_bytes == 0:
            return
        self._full_warning_logged = True
        max_gb = self._max_size_bytes / (1024**3)
        logging.warning(
            "SequenceInferenceCache is full (%.4g GB). "
            "Further cache misses will not be stored until clear().",
            max_gb,
        )

    def _assemble_and_restore(self, per_index_rows: List[Optional[Tuple[Any, ...]]]) -> Any:
        merged = self._assemble_batch_output(per_index_rows)
        if self._output_specs is None:
            return merged
        return self._restore_batch_output(merged, self._output_specs)

    @staticmethod
    def _to_cache_value(value: Any) -> Any:
        """Store tensors/arrays on CPU; pass through None and non-numeric types."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return np.asarray(value)
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    @staticmethod
    def _get_output_spec(value: Any) -> OutputSpec:
        if value is None:
            return ("none",)
        if isinstance(value, np.ndarray):
            return ("numpy", value.dtype)
        if torch is not None and isinstance(value, torch.Tensor):
            return ("torch", value.device, value.dtype)
        if isinstance(value, list):
            return ("list",)
        return ("raw",)

    @staticmethod
    def _restore_value(value: Any, spec: OutputSpec) -> Any:
        kind = spec[0]
        if kind == "none" or value is None:
            return None
        if kind == "numpy":
            return value
        if kind == "torch":
            if torch is None:
                return value
            _, device, dtype = spec
            tensor = torch.from_numpy(value)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            return tensor.to(device)
        if kind == "list":
            return value
        return value

    @classmethod
    def _restore_batch_output(
        cls, merged: Any, output_specs: Tuple[OutputSpec, ...]
    ) -> Any:
        if not isinstance(merged, tuple):
            return cls._restore_value(merged, output_specs[0])
        restored = tuple(
            cls._restore_value(slot, spec) for slot, spec in zip(merged, output_specs)
        )
        if len(restored) == 1:
            return restored[0]
        return restored

    def _make_key(self, sequence: str, extra_args: Sequence[Any]) -> Hashable:
        return (sequence,) + tuple(self._normalize_for_key(arg) for arg in extra_args)

    def _normalize_for_key(self, arg: Any) -> Hashable:
        if arg is None:
            return None
        if isinstance(arg, (bool, int, float, str)):
            return arg
        if isinstance(arg, np.ndarray):
            return (arg.shape, arg.tobytes())
        if torch is not None and isinstance(arg, torch.Tensor):
            arr = arg.detach().cpu().numpy()
            return (arr.shape, arr.tobytes())
        if isinstance(arg, (list, tuple)):
            return tuple(self._normalize_for_key(item) for item in arg)
        return repr(arg)

    def _is_batch_aligned(self, arg: Any, batch_size: int) -> bool:
        if isinstance(arg, list) and len(arg) == batch_size:
            return True
        if isinstance(arg, np.ndarray) and arg.ndim > 0 and arg.shape[0] == batch_size:
            return True
        if torch is not None and isinstance(arg, torch.Tensor) and arg.ndim > 0 and arg.shape[0] == batch_size:
            return True
        return False

    def _filter_args(
        self,
        sequences: List[str],
        extra_args: Sequence[Any],
        indices: List[int],
        batch_size: int,
    ) -> Tuple[Any, ...]:
        filtered_sequences = [sequences[i] for i in indices]
        filtered_extra = []
        for arg in extra_args:
            if self._is_batch_aligned(arg, batch_size):
                if isinstance(arg, list):
                    filtered_extra.append([arg[i] for i in indices])
                elif isinstance(arg, np.ndarray):
                    filtered_extra.append(arg[indices])
                else:
                    filtered_extra.append(arg[indices])
            else:
                filtered_extra.append(arg)
        return (filtered_sequences, *filtered_extra)

    @staticmethod
    def _slice_row(output: Any, idx: int) -> Any:
        if isinstance(output, np.ndarray):
            return output[idx]
        if torch is not None and isinstance(output, torch.Tensor):
            return output[idx]
        if isinstance(output, list):
            return output[idx]
        return output

    def _assemble_batch_output(self, per_index_rows: List[Optional[Tuple[Any, ...]]]) -> Any:
        batch_size = len(per_index_rows)
        num_outputs = len(per_index_rows[0])  # type: ignore[index]
        merged_slots = []
        for out_idx in range(num_outputs):
            rows = [per_index_rows[i][out_idx] for i in range(batch_size)]  # type: ignore[index]
            merged_slots.append(self._merge_rows(rows))
        if num_outputs == 1:
            return merged_slots[0]
        return tuple(merged_slots)

    @staticmethod
    def _pad_embedding_rows(rows: List[np.ndarray]) -> np.ndarray:
        """Pad variable-length per-sequence embedding rows to a common seq length."""
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        max_len = max(row.shape[0] for row in rows)
        hidden_dim = rows[0].shape[1]
        out = np.zeros((len(rows), max_len, hidden_dim), dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            out[i, : row.shape[0], :] = row
        return out

    @staticmethod
    def _pad_prob_rows(rows: List[np.ndarray]) -> np.ndarray:
        """Pad variable-length per-sequence 1D rows (e.g. token probs) to a common length."""
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        max_len = max(row.shape[0] for row in rows)
        out = np.zeros((len(rows), max_len), dtype=rows[0].dtype)
        for i, row in enumerate(rows):
            out[i, : row.shape[0]] = row
        return out

    @staticmethod
    def _merge_rows(rows: List[Any]) -> Any:
        if all(row is None for row in rows):
            return None

        first = next(row for row in rows if row is not None)

        if isinstance(first, np.ndarray) and first.ndim == 2:
            padded = [
                np.zeros_like(first) if row is None else np.asarray(row) for row in rows
            ]
            return SequenceInferenceCache._pad_embedding_rows(padded)

        if isinstance(first, np.ndarray) and first.ndim == 1:
            max_len = max(np.asarray(row).shape[0] for row in rows if row is not None)
            padded = [
                np.zeros(max_len, dtype=first.dtype) if row is None else np.asarray(row)
                for row in rows
            ]
            return SequenceInferenceCache._pad_prob_rows(padded)

        if isinstance(first, np.ndarray):
            out = np.zeros((len(rows),) + first.shape, dtype=first.dtype)
            for i, row in enumerate(rows):
                if row is not None:
                    out[i] = row
            return out

        if isinstance(first, list):
            return rows
        return rows
