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
E2E download tests: verify tasks fetch data into an empty data directory.

Simulates a fresh user with no local benchmark cache. Requires network access.

Run:
  pytest tests/e2e/test_download.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import pytest

from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from gfmbench_api.tasks.concrete.lrb_cage_task import LRBCagePredictionTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from usage_examples.run_benchmark import TASK_REGISTRY

EXCLUDED_DOWNLOAD_TASKS = frozenset({"vepeval_clinvar"})
DOWNLOAD_TASKS: dict[str, Type] = {
    name: cls
    for name, cls in TASK_REGISTRY.items()
    if name not in EXCLUDED_DOWNLOAD_TASKS
}
# CAGE is download-tested directly until the benchmark runner supports regression.
DOWNLOAD_TASKS["lrb_cage_prediction"] = LRBCagePredictionTask

# TraitGym downloads HF data but expects hg38.fa to already exist under data_root.
TASKS_NEEDING_PREFETCHED_REFERENCE_GENOME = frozenset(
    {"traitgym_complex", "traitgym_mendelian"}
)

# LOL-EVE loads via hf_hub_download into the HF cache, not under data_root.
TASKS_WITHOUT_LOCAL_DATA_DIR = frozenset({"loleve_causal_eqtl"})

DOWNLOAD_TASK_CONFIG = {
    "max_num_samples": 4,
    "batch_size": 2,
    "max_sequence_length": 256,
}


@pytest.fixture(scope="module")
def download_data_root(tmp_path_factory) -> Path:
    """Empty data root shared across tests (hg38 is downloaded at most once)."""
    root = tmp_path_factory.mktemp("download_data")
    assert not any(root.iterdir()), "expected empty data directory"
    return root


def _assert_download_artifacts(data_root: Path, task_name: str) -> None:
    """Verify on-disk artifacts under data_root are present and non-empty."""
    task_dir = data_root / task_name
    if task_name not in TASKS_WITHOUT_LOCAL_DATA_DIR:
        assert task_dir.is_dir(), f"Expected task data directory {task_dir}"
        files = [path for path in task_dir.rglob("*") if path.is_file()]
        assert files, f"Expected at least one downloaded file under {task_dir}"
        empty = [
            str(path.relative_to(task_dir))
            for path in files
            if path.stat().st_size == 0
        ]
        assert not empty, f"Empty downloaded files under {task_dir}: {empty}"

    ref_genome = data_root / "reference_genome" / "hg38.fa"
    if ref_genome.is_file():
        assert ref_genome.stat().st_size > 0, f"Reference genome is empty: {ref_genome}"


@pytest.mark.parametrize(
    ("task_name", "task_cls"),
    sorted(DOWNLOAD_TASKS.items()),
    ids=sorted(DOWNLOAD_TASKS.keys()),
)
def test_task_downloads_from_empty_data_dir(
    download_data_root: Path,
    task_name: str,
    task_cls: Type,
):
    """Task init should download data and expose non-empty on-disk artifacts and splits."""
    if task_name in TASKS_NEEDING_PREFETCHED_REFERENCE_GENOME:
        ensure_reference_genome(str(download_data_root / "reference_genome" / "hg38.fa"))

    task = task_cls(
        root_data_dir_path=str(download_data_root),
        task_config=dict(DOWNLOAD_TASK_CONFIG),
    )

    assert task.get_task_name() == task_name
    _assert_download_artifacts(download_data_root, task_name)
    assert len(task.test_dataset) > 0

    attrs = task.get_task_attributes()
    if attrs.get("has_finetuning_data"):
        finetune = task.get_finetune_dataset()
        assert finetune is not None and len(finetune) > 0


def test_gue_download_is_idempotent(download_data_root: Path):
    """Second init should reuse cached files without error."""
    kwargs = {
        "root_data_dir_path": str(download_data_root),
        "task_config": dict(DOWNLOAD_TASK_CONFIG),
    }
    task1 = GuePromoterAllTask(**kwargs)
    task_dir = download_data_root / "gue_promoter_all"
    _assert_download_artifacts(download_data_root, "gue_promoter_all")

    csv_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in task_dir.glob("*.csv")
    }

    task2 = GuePromoterAllTask(**kwargs)
    _assert_download_artifacts(download_data_root, "gue_promoter_all")
    for name, mtime_ns in csv_mtimes.items():
        assert (task_dir / name).stat().st_mtime_ns == mtime_ns

    assert len(task2.test_dataset) == len(task1.test_dataset)
