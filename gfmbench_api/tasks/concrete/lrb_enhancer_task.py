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

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import (
    BaseGFMSupervisedSingleSeqTask,
)
from gfmbench_api.utils.lrb_local import build_regulatory_examples


class LRBEnhancerTask(BaseGFMSupervisedSingleSeqTask):
    """LRB Regulatory Elements - Enhancer (binary classification).

    A 200bp genomic bin is positive if it overlaps an annotated enhancer
    cis-regulatory element (SCREEN/ENCODE) by >=50%, negative otherwise. Each
    bin is centered+padded to ``sequence_length`` bp against GRCh38; the label
    is binary. Train = chr1-7,10-22; test = chr8,9.
    """

    def _get_default_max_seq_len(self) -> int:
        # Builder default context is 100kb.
        return 100000

    def _get_num_labels(self) -> int:
        return 1

    def _get_num_classes(self) -> int:
        return 2

    def get_task_name(self) -> str:
        return "lrb_regulatory_element_enhancer"

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        cache_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        genome_dir = os.path.join(self.root_data_dir_path, "reference_genome")
        subset = bool(self.task_config.get("subset", True))
        logging.info(
            "Building LRB enhancer (sequence_length=%d, subset=%s)...",
            self.max_sequence_length,
            subset,
        )
        train_dataset, test_dataset = build_regulatory_examples(
            cache_dir=cache_dir,
            genome_dir=genome_dir,
            sequence_length=self.max_sequence_length,
            element="enhancer",
            subset=subset,
            max_num_samples=self.max_num_samples,
        )
        return train_dataset, None, test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
