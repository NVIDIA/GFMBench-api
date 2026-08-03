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
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import (
    BaseGFMSupervisedSingleSeqTask,
)
from gfmbench_api.utils.lrb_local import build_chromatin_examples


class LRBHistoneMarksTask(BaseGFMSupervisedSingleSeqTask):
    """LRB Chromatin Features - Histone Marks (multi-label classification).

    DeepSea-derived: each 200bp bin carries a binary vector over 20 histone-mark
    tracks (positive if the bin overlaps a peak of that profile by >50%). Each
    bin is centered in a ``sequence_length`` bp window against GRCh37/hg19; the
    label is a 20-length binary vector. Train = chr1-7,10-22; test = chr8,9.
    """

    # DeepSea histone subset used by LRB; the actual count is read from the data
    # in _create_datasets, this is only the fallback before materialization.
    DEFAULT_NUM_TRACKS = 20

    def _get_default_max_seq_len(self) -> int:
        # Builder requires >=200bp.
        return 100000

    def _get_num_labels(self) -> int:
        return getattr(self, "_num_labels", self.DEFAULT_NUM_TRACKS)

    def _get_num_classes(self) -> int:
        return 2

    def get_task_name(self) -> str:
        return "lrb_chromatin_features_histone_marks"

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        cache_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        genome_dir = os.path.join(self.root_data_dir_path, "reference_genome")
        subset = bool(self.task_config.get("subset", True))
        logging.info(
            "Building LRB histone marks (sequence_length=%d, subset=%s)...",
            self.max_sequence_length,
            subset,
        )
        train_dataset, test_dataset = build_chromatin_examples(
            cache_dir=cache_dir,
            genome_dir=genome_dir,
            sequence_length=self.max_sequence_length,
            feature="histone",
            subset=subset,
            max_num_samples=self.max_num_samples,
        )
        sample = train_dataset[0] if train_dataset else (test_dataset[0] if test_dataset else None)
        if sample is not None:
            self._num_labels = int(sample[1].shape[0])
        return train_dataset, None, test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
