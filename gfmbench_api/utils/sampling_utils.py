# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic sample limiting that preserves label diversity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def diverse_sample_indices(
    labels: Sequence,
    max_samples: int,
    *,
    seed: int = 0,
    min_per_class: int = 2,
) -> np.ndarray:
    """Select at most ``max_samples`` rows while retaining observed label values.

    Scalar labels are treated as single-label classification. Vector labels are
    treated as multilabel targets; the selection attempts to retain
    ``min_per_class`` rows for every observed value of every target.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if min_per_class <= 0:
        raise ValueError("min_per_class must be positive")

    label_rows = [np.asarray(label).reshape(-1) for label in labels]
    if not label_rows:
        return np.empty(0, dtype=np.int64)
    widths = {row.size for row in label_rows}
    if len(widths) != 1:
        raise ValueError("All labels must have the same number of targets")

    num_rows = len(label_rows)
    if num_rows <= max_samples:
        return np.arange(num_rows, dtype=np.int64)

    label_matrix = np.stack(label_rows)
    requirements: dict[tuple[int, object], int] = {}
    row_features: list[set[tuple[int, object]]] = []
    for row in label_matrix:
        features = {
            (target_idx, value.item() if hasattr(value, "item") else value)
            for target_idx, value in enumerate(row)
        }
        row_features.append(features)

    for target_idx in range(label_matrix.shape[1]):
        values, counts = np.unique(label_matrix[:, target_idx], return_counts=True)
        for value, count in zip(values, counts):
            key = (target_idx, value.item() if hasattr(value, "item") else value)
            requirements[key] = min(min_per_class, int(count))

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(num_rows)
    selected: list[int] = []
    selected_set: set[int] = set()

    while any(remaining > 0 for remaining in requirements.values()):
        best_idx = None
        best_score = 0
        for idx in tie_order:
            idx = int(idx)
            if idx in selected_set:
                continue
            score = sum(
                requirements.get(feature, 0) > 0
                for feature in row_features[idx]
            )
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        if len(selected) >= max_samples:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        for feature in row_features[best_idx]:
            if requirements.get(feature, 0) > 0:
                requirements[feature] -= 1

    remaining_slots = max_samples - len(selected)
    if remaining_slots:
        remaining = np.array(
            [idx for idx in range(num_rows) if idx not in selected_set],
            dtype=np.int64,
        )
        selected.extend(
            rng.choice(remaining, size=remaining_slots, replace=False).tolist()
        )

    return rng.permutation(np.asarray(selected, dtype=np.int64))


def diverse_sample_dataframe(
    df: pd.DataFrame,
    labels: Sequence,
    max_samples: int | None,
    *,
    seed: int = 0,
    min_per_class: int = 2,
) -> pd.DataFrame:
    """Return a row-aligned, diversity-preserving subset of ``df``."""
    if max_samples is None or len(df) <= max_samples:
        return df.reset_index(drop=True)
    if len(labels) != len(df):
        raise ValueError("labels must contain one entry per dataframe row")
    indices = diverse_sample_indices(
        labels,
        max_samples,
        seed=seed,
        min_per_class=min_per_class,
    )
    return df.iloc[indices].reset_index(drop=True)
