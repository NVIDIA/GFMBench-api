# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from gfmbench_api.utils.sampling_utils import (
    diverse_sample_dataframe,
    diverse_sample_indices,
)


def test_scalar_sampling_is_deterministic_and_proportionally_stratified():
    labels = np.array([0] * 90 + [1] * 10)

    first = diverse_sample_indices(labels, 20, seed=7)
    second = diverse_sample_indices(labels, 20, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first)) == 20
    np.testing.assert_array_equal(np.bincount(labels[first]), [18, 2])


def test_different_seeds_change_selected_rows():
    labels = np.array([0] * 20 + [1] * 20)

    first = diverse_sample_indices(labels, 10, seed=1)
    second = diverse_sample_indices(labels, 10, seed=2)

    assert set(first) != set(second)


def test_multitarget_sampling_retains_every_observed_value():
    labels = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        * 2
    )

    indices = diverse_sample_indices(labels, 4, min_per_class=1)
    sampled = labels[indices]

    assert len(indices) == 4
    for target_idx in range(labels.shape[1]):
        assert set(sampled[:, target_idx]) == {0, 1}


def test_minimum_coverage_includes_all_available_rare_class_rows():
    labels = np.array([0, 0, 0, 1])

    indices = diverse_sample_indices(labels, 3, min_per_class=2)

    np.testing.assert_array_equal(np.bincount(labels[indices]), [2, 1])


def test_infeasible_minimum_coverage_raises():
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="need at least 4, got 3"):
        diverse_sample_indices(labels, 3, min_per_class=2)


def test_dataframe_noop_returns_same_object_and_preserves_index():
    df = pd.DataFrame({"label": [0, 1]}, index=[10, 20])

    result = diverse_sample_dataframe(df, df["label"], max_samples=None)

    assert result is df
    assert result.index.tolist() == [10, 20]


def test_dataframe_validates_label_alignment_even_when_sampling_is_noop():
    df = pd.DataFrame({"label": [0, 1]})

    with pytest.raises(ValueError, match="one entry per dataframe row"):
        diverse_sample_dataframe(df, [0], max_samples=None)


def test_dataframe_sampling_keeps_rows_aligned_and_resets_index():
    df = pd.DataFrame(
        {"label": [0, 0, 1, 1], "value": ["a", "b", "c", "d"]},
        index=[10, 20, 30, 40],
    )

    result = diverse_sample_dataframe(
        df, df["label"], max_samples=2, min_per_class=1
    )

    assert result.index.tolist() == [0, 1]
    assert set(result["label"]) == {0, 1}
    assert all(
        value in {"a", "b"} if label == 0 else value in {"c", "d"}
        for label, value in zip(result["label"], result["value"])
    )


@pytest.mark.parametrize("max_samples", [0, -1])
def test_nonpositive_max_samples_raises(max_samples):
    with pytest.raises(ValueError, match="max_samples must be positive"):
        diverse_sample_indices([0, 1], max_samples)


def test_nonpositive_minimum_raises():
    with pytest.raises(ValueError, match="min_per_class must be positive"):
        diverse_sample_indices([0, 1], 1, min_per_class=0)


def test_inconsistent_target_widths_raise():
    with pytest.raises(ValueError, match="same number of targets"):
        diverse_sample_indices([[0, 1], [1]], 1)


def test_empty_labels_return_empty_indices():
    indices = diverse_sample_indices([], 1)

    assert indices.dtype == np.int64
    assert indices.size == 0
