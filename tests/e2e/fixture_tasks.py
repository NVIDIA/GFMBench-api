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

"""Minimal task implementations that load local fixture CSVs (no network downloads)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_classification_task import (
    BaseGFMSupervisedClassificationTask,
)
from gfmbench_api.tasks.base.base_gfm_zeroshot_general_indel_task import (
    BaseGFMZeroShotGeneralIndelTask,
)
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends

FIXTURE_VARIANT_DIR = "fixture_variant"


def _variant_data_dir(root_data_dir_path: str) -> str:
    return os.path.join(root_data_dir_path, FIXTURE_VARIANT_DIR)


def _load_variant_split_csv(data_dir: str, split: str, max_samples: Optional[int]) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(path)
    if max_samples is not None:
        df = df.head(min(max_samples, len(df)))
    return df


def _build_variant_dataset(
    df: pd.DataFrame,
    max_sequence_length: int,
) -> list:
    rows = []
    for var_seq, ref_seq, label in zip(
        df["variant_sequence"],
        df["reference_sequence"],
        df["label"],
    ):
        var_seq = truncate_sequence_from_ends(str(var_seq), max_sequence_length)
        ref_seq = truncate_sequence_from_ends(str(ref_seq), max_sequence_length)
        rows.append((var_seq, ref_seq, int(label), np.array([])))
    return rows


class FixtureSupervisedVariantTask(BaseGFMSupervisedClassificationTask):
    """Supervised variant-effect task backed by shared ``fixture_variant/`` CSVs."""

    classification_mode = "single_label"
    input_structure = "variant_reference_pair"

    def get_task_name(self) -> str:
        return "fixture_supervised_variant"

    def _get_default_max_seq_len(self) -> int:
        return 64

    def _get_num_classes(self) -> int:
        return 2

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        data_dir = _variant_data_dir(self.root_data_dir_path)
        train_df = _load_variant_split_csv(data_dir, "train", self.max_num_samples)
        test_df = _load_variant_split_csv(data_dir, "test", self.max_num_samples)
        train_dataset = _build_variant_dataset(train_df, self.max_sequence_length)
        test_dataset = _build_variant_dataset(test_df, self.max_sequence_length)
        return train_dataset, None, test_dataset

    def get_conditional_input_meta_data_frame(self) -> None:
        return None


class FixtureZeroShotSNVTask(BaseGFMZeroShotSNVTask):
    """Zero-shot SNV task backed by shared ``fixture_variant/test.csv``."""

    SNV_POSITION = 16

    def get_task_name(self) -> str:
        return "fixture_zero_shot_snv"

    def _get_default_max_seq_len(self) -> int:
        return 64

    def use_reference_cache(self) -> bool:
        return False

    def _get_variant_position_in_sequence(self) -> int:
        return self.SNV_POSITION

    def _create_test_dataset(self) -> list:
        data_dir = _variant_data_dir(self.root_data_dir_path)
        test_df = _load_variant_split_csv(data_dir, "test", self.max_num_samples)
        return _build_variant_dataset(test_df, self.max_sequence_length)

    def get_conditional_input_meta_data_frame(self) -> None:
        return None


class FixtureZeroShotIndelTask(BaseGFMZeroShotGeneralIndelTask):
    """Zero-shot indel task backed by shared ``fixture_variant/test.csv``."""

    def get_task_name(self) -> str:
        return "fixture_zero_shot_indel"

    def _get_default_max_seq_len(self) -> int:
        return 64

    def use_reference_cache(self) -> bool:
        return False

    def _create_test_dataset(self) -> list:
        data_dir = _variant_data_dir(self.root_data_dir_path)
        test_df = _load_variant_split_csv(data_dir, "test", self.max_num_samples)
        return _build_variant_dataset(test_df, self.max_sequence_length)

    def get_conditional_input_meta_data_frame(self) -> None:
        return None
