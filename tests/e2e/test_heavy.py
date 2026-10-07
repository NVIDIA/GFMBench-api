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
Heavy E2E regression: run DNABERT2 on all tasks in sanity mode, compare to baseline.

Regression (included in ``pytest tests/``):
  pytest tests/e2e/test_heavy.py::test_heavy_sanity_regression

Refresh pinned baseline (run explicitly by name):
  pytest tests/e2e/test_heavy.py::test_heavy_update_baseline -s
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gfmbench_api.tasks.base.base_gfm_supervised_classification_task import (
    BaseGFMSupervisedClassificationTask,
)
from gfmbench_api.tasks.base.base_gfm_zero_shot_task import BaseGFMZeroShotTask
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import (
    BaseGFMZeroShotSNVTask,
)
from tests.e2e.baseline_utils import (
    compare_to_baseline,
    format_failures,
    load_baseline,
    load_results,
)
from usage_examples.run_benchmark import TASK_REGISTRY
from usage_examples.run_benchmark import main as run_benchmark_main

DEFAULT_HEAVY_MODEL = "DNABERT2"
DEFAULT_ATOL = 0.005

EXCLUDED_HEAVY_TASKS = frozenset({"vepeval_clinvar"})


def _expected_auroc_metrics(task_cls: type) -> set[str]:
    if issubclass(task_cls, BaseGFMSupervisedClassificationTask):
        return {"classification_auroc"}
    if issubclass(task_cls, BaseGFMZeroShotTask):
        metrics = {
            "sequence_embeddings_cosinesim_auroc",
            "sequence_embeddings_l2_auroc",
        }
        if issubclass(task_cls, BaseGFMZeroShotSNVTask):
            metrics.update({
                "snv_variant_effect_cosinesim_auroc",
                "snv_variant_effect_prediction_masked_llr_auroc",
            })
        return metrics
    return set()


def _assert_all_auroc_metrics_are_finite(results_df: pd.DataFrame) -> None:
    finite_results = results_df.copy()
    finite_results["actual"] = pd.to_numeric(
        finite_results["actual"], errors="coerce"
    )
    finite_results = finite_results[
        np.isfinite(finite_results["actual"].to_numpy(dtype=float))
    ]
    finite_keys = set(zip(finite_results["task"], finite_results["metric"]))

    missing = []
    for task_name, task_cls in TASK_REGISTRY.items():
        if task_name in EXCLUDED_HEAVY_TASKS:
            continue
        for metric in _expected_auroc_metrics(task_cls):
            if (task_name, metric) not in finite_keys:
                missing.append(f"{task_name} / {metric}")

    assert not missing, (
        "Sanity sampling did not produce finite AUROC metrics:\n  "
        + "\n  ".join(sorted(missing))
    )


def _baseline_path(model_name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / f"{safe}_sanity_baseline.csv"
    )


def _benchmark_argv(
    *,
    root_data_dir_path: Path,
    csv_path: Path,
    model_name: str = DEFAULT_HEAVY_MODEL,
) -> list[str]:
    return [
        "--root_data_dir_path",
        str(root_data_dir_path),
        "--csv_path",
        str(csv_path),
        "--report_algo_name",
        f"{model_name.lower()}_heavy",
        "--model",
        model_name,
        "--linear_prob",
        "--epochs",
        "3",
        "--sanity_check_mode",
        "--disable_safe_model_call",
        "--seed",
        "0",
        "--exclude_tasks",
        *EXCLUDED_HEAVY_TASKS,
    ]


@pytest.fixture(scope="module")
def heavy_data_root(request, tmp_path_factory) -> Path:
    """Shared data directory; task init downloads datasets as needed.

    Pass --heavy-data-root PATH to reuse a persistent directory across runs
    (avoids re-downloading the reference genome and task datasets every time).
    Falls back to a fresh temp directory when the option is not provided.
    """
    persistent = request.config.getoption("--heavy-data-root")
    if persistent:
        path = Path(persistent)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("heavy_data")


def _run_heavy_benchmark(
    heavy_data_root: Path,
    csv_path: Path,
    model_name: str = DEFAULT_HEAVY_MODEL,
) -> pd.DataFrame:
    run_benchmark_main(_benchmark_argv(
        root_data_dir_path=heavy_data_root,
        csv_path=csv_path,
        model_name=model_name,
    ))
    return load_results(csv_path)


def _write_baseline(results_df: pd.DataFrame, baseline_path: Path) -> None:
    baseline_rows = []
    for _, row in results_df.iterrows():
        actual = pd.to_numeric(row["actual"], errors="coerce")
        if not np.isfinite(actual):
            continue
        baseline_rows.append(
            {
                "task": row["task"],
                "metric": row["metric"],
                "expected": actual,
                "atol": DEFAULT_ATOL,
            }
        )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)


def test_heavy_sanity_regression(heavy_data_root, tmp_path):
    """Run DNABERT2 on all tasks (sanity mode) and compare to the pinned baseline."""
    csv_path = tmp_path / "heavy_results.csv"
    baseline_path = _baseline_path(DEFAULT_HEAVY_MODEL)

    if not baseline_path.is_file():
        pytest.fail(
            f"Missing baseline {baseline_path}. "
            "Create it with: "
            "pytest tests/e2e/test_heavy.py::test_heavy_update_baseline -s"
        )

    results_df = _run_heavy_benchmark(heavy_data_root, csv_path)
    _assert_all_auroc_metrics_are_finite(results_df)
    baseline_df = load_baseline(baseline_path)
    failures, warnings = compare_to_baseline(
        results_df,
        baseline_df,
        default_atol=DEFAULT_ATOL,
    )

    if warnings:
        print("\n".join(warnings))

    assert not failures, "Metric drift vs baseline:\n" + format_failures(failures)


def test_heavy_update_baseline(heavy_data_root, tmp_path):
    """Run DNABERT2 on all tasks (sanity mode) and write the pinned baseline CSV."""
    csv_path = tmp_path / "heavy_results.csv"
    baseline_path = _baseline_path(DEFAULT_HEAVY_MODEL)
    results_df = _run_heavy_benchmark(heavy_data_root, csv_path)
    _assert_all_auroc_metrics_are_finite(results_df)
    _write_baseline(results_df, baseline_path)
    print(f"Wrote updated baseline to {baseline_path}")
