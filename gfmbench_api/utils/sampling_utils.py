# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, stratified sample limiting that preserves label diversity."""

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
    treated as multiple targets. For every observed value of every target, the
    selection guarantees ``min(min_per_class, available_rows)`` rows, or raises
    ``ValueError`` when that coverage cannot fit in ``max_samples``. Remaining
    capacity is filled toward proportional per-target class quotas.
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
    encoded_targets: list[np.ndarray] = []
    class_counts: list[np.ndarray] = []
    minimum_requirements: list[np.ndarray] = []
    stratified_quotas: list[np.ndarray] = []
    for target_idx in range(label_matrix.shape[1]):
        _, encoded, counts = np.unique(
            label_matrix[:, target_idx],
            return_inverse=True,
            return_counts=True,
        )
        minimum = np.minimum(counts, min_per_class).astype(np.int64)
        if int(minimum.sum()) > max_samples:
            raise ValueError(
                "max_samples is too small to retain the requested "
                f"min_per_class coverage for target {target_idx}: "
                f"need at least {int(minimum.sum())}, got {max_samples}"
            )

        quota = minimum.copy()
        desired = max_samples * counts / num_rows
        for _ in range(max_samples - int(quota.sum())):
            available = quota < counts
            deficits = desired - quota
            deficits[~available] = -np.inf
            quota[int(np.argmax(deficits))] += 1

        encoded_targets.append(encoded)
        class_counts.append(counts)
        minimum_requirements.append(minimum)
        stratified_quotas.append(quota)

    encoded_matrix = np.column_stack(encoded_targets)
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(num_rows)
    selected: list[int] = []
    selected_mask = np.zeros(num_rows, dtype=bool)

    def select_toward(requirements: list[np.ndarray]) -> None:
        while len(selected) < max_samples and any(
            np.any(remaining > 0) for remaining in requirements
        ):
            scores = np.zeros(num_rows, dtype=np.int64)
            for target_idx, remaining in enumerate(requirements):
                scores += remaining[encoded_matrix[:, target_idx]] > 0
            scores[selected_mask] = -1
            best_idx = int(tie_order[np.argmax(scores[tie_order])])
            if scores[best_idx] <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for target_idx, remaining in enumerate(requirements):
                class_idx = encoded_matrix[best_idx, target_idx]
                if remaining[class_idx] > 0:
                    remaining[class_idx] -= 1

    remaining_minimums = [
        requirement.copy() for requirement in minimum_requirements
    ]
    select_toward(remaining_minimums)
    if any(np.any(remaining > 0) for remaining in remaining_minimums):
        raise ValueError(
            "Unable to satisfy min_per_class for every target within "
            f"max_samples={max_samples}"
        )

    selected_counts = [
        np.bincount(
            encoded_matrix[selected, target_idx],
            minlength=len(class_counts[target_idx]),
        )
        for target_idx in range(encoded_matrix.shape[1])
    ]
    remaining_quotas = [
        np.maximum(quota - counts, 0)
        for quota, counts in zip(stratified_quotas, selected_counts)
    ]
    select_toward(remaining_quotas)

    remaining_slots = max_samples - len(selected)
    if remaining_slots:
        remaining = np.flatnonzero(~selected_mask)
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
    if len(labels) != len(df):
        raise ValueError("labels must contain one entry per dataframe row")
    if max_samples is None or len(df) <= max_samples:
        return df
    indices = diverse_sample_indices(
        labels,
        max_samples,
        seed=seed,
        min_per_class=min_per_class,
    )
    return df.iloc[indices].reset_index(drop=True)
