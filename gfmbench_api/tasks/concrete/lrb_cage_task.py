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

# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark — CC-BY-NC-4.0
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_regression_task import (
    BaseGFMSupervisedRegressionTask,
    OutputSpatiality,
)
from gfmbench_api.utils.lrb_local import build_cage_examples


class LRBCagePredictionTask(BaseGFMSupervisedRegressionTask):
    """LRB CAGE Prediction (binned multi-track regression).

    Predicts FANTOM5 CAGE expression (50 tracks, via Basenji/Enformer) in 128bp
    bins. Each example yields a ``[num_bins, 50]`` target matrix where
    ``num_bins = sequence_length / 128`` and ``sequence_length`` is an even
    multiple of 128. Targets are raw counts, so we ``log1p`` them and
    train/evaluate in log space. Reference genome is GRCh38.

    The dataset has a validation split; following the dataset card we merge it
    into the training set.
    """

    BIN_SIZE_BP = 128
    BUILDER_MAX_LENGTH = 114688  # 896 bins x 128bp (full Basenji/Enformer window)
    DEFAULT_NUM_TRACKS = 50
    def _get_default_max_seq_len(self) -> int:
        return self.BUILDER_MAX_LENGTH

    def _get_num_labels(self) -> int:
        return getattr(self, "_num_tracks", self.DEFAULT_NUM_TRACKS)

    def _get_output_spatiality(self) -> OutputSpatiality:
        return OutputSpatiality.BINNED

    def get_task_name(self) -> str:
        return "lrb_cage_prediction"

    def _resolve_cage_sequence_length(self) -> int:
        """Round the effective window down to an even multiple of 128bp.

        Labels are symmetric 128bp bins (``(num_bins) % 2 == 0`` upstream), so a
        model max length such as 2500 must be snapped down to a valid window.
        """
        bins = self.max_sequence_length // self.BIN_SIZE_BP
        if bins < 2:
            raise ValueError(
                f"CAGE requires at least 2 bins (256bp); effective max length "
                f"{self.max_sequence_length} yields {bins} bins."
            )
        if bins % 2 != 0:
            bins -= 1
        self._num_bins = bins
        return bins * self.BIN_SIZE_BP

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        cache_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        genome_dir = os.path.join(self.root_data_dir_path, "reference_genome")
        seq_len = self._resolve_cage_sequence_length()
        logging.info(
            "Building LRB CAGE (sequence_length=%d, num_bins=%d)...",
            seq_len,
            self._num_bins,
        )
        train_examples, validation_examples, test_examples = build_cage_examples(
            cache_dir=cache_dir,
            genome_dir=genome_dir,
            sequence_length=seq_len,
            max_num_samples=self.max_num_samples,
        )
        # Merge validation into train (per the dataset card).
        train_examples = train_examples + validation_examples

        sample = train_examples[0] if train_examples else (test_examples[0] if test_examples else None)
        if sample is not None:
            self._num_tracks = int(sample[1].shape[-1])
        return train_examples, None, test_examples

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
