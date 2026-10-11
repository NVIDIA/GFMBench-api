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
# - https://github.com/MAGICS-LAB/DNABERT_2 — Apache-2.0
import logging
import os
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import BaseGFMSupervisedSingleSeqTask
import numpy as np
from gfmbench_api.utils.fileutils import (
    ensure_verified_download,
    materialize_csv_splits_from_zip,
)
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends
from gfmbench_api.utils.sampling_utils import diverse_sample_dataframe

GUE_ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-&export=download&confirm=t"
)
GUE_MANUAL_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view"
)
GUE_ARCHIVE_SHA256 = "581ba5a69843769f4cbc34470e2fe970f8ee371280914bd66574af01ee60cd79"
GUE_PROMOTER_ARCHIVE_DIR = "GUE/prom/prom_300_all"


class GuePromoterAllTask(BaseGFMSupervisedSingleSeqTask):
    """GUE promoter prediction task (binary classification)."""
    
    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length (300bp)."""
        return 300

    def _get_num_labels(self) -> int:
        return 1
    
    def _get_num_classes(self):
        """Return 2 (binary: promoter vs non-promoter)."""
        return 2

    def get_task_name(self):
        """Return task name (identical to data directory name)."""
        return "gue_promoter_all"

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        """Create train, validation, and test datasets from CSV files."""
        data_dir = os.path.join(self.root_data_dir_path, self._get_task_data_dir_name())
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "dev.csv")
        test_path = os.path.join(data_dir, "test.csv")

        # Download data if not exists
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            logging.info("Downloading %s from the canonical GUE archive...", self.get_task_name())
            archive_path = os.path.join(
                self.root_data_dir_path, ".downloads", "GUE.zip"
            )
            try:
                ensure_verified_download(
                    url=GUE_ARCHIVE_URL,
                    destination_path=archive_path,
                    expected_sha256=GUE_ARCHIVE_SHA256,
                    description="canonical GUE archive from the DNABERT-2 source",
                )
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "Failed to automatically download the canonical GUE archive, "
                    "possibly because the Google Drive download quota was exceeded.\n\n"
                    "To download it manually:\n"
                    f"1. Visit: {GUE_MANUAL_DOWNLOAD_URL}\n"
                    "2. Download the ZIP archive.\n"
                    f"3. Save it as: {archive_path}\n"
                    f"4. Verify its SHA-256 is: {GUE_ARCHIVE_SHA256}\n\n"
                    f"Original error: {error}"
                ) from error
            materialize_csv_splits_from_zip(
                archive_path=archive_path,
                member_directories=[GUE_PROMOTER_ARCHIVE_DIR],
                splits=["train", "test", "dev"],
                local_dir=data_dir,
            )
            logging.info(f"Data saved to: {data_dir}")

        # Load CSVs
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Limit samples if max_num_samples is specified
        if self.max_num_samples is not None:
            train_df = diverse_sample_dataframe(
                train_df, train_df["label"], self.max_num_samples
            )
            val_df = diverse_sample_dataframe(
                val_df, val_df["label"], self.max_num_samples
            )
            test_df = diverse_sample_dataframe(
                test_df, test_df["label"], self.max_num_samples
            )

        # Keep sequences as strings; model handles tokenization
        # Truncate from ends if needed (preserving center), using self.max_sequence_length
        X_train = [truncate_sequence_from_ends(seq, self.max_sequence_length) 
                   for seq in train_df['sequence']]
        y_train = torch.tensor(train_df['label'].values, dtype=torch.long)

        X_val = [truncate_sequence_from_ends(seq, self.max_sequence_length) 
                 for seq in val_df['sequence']]
        y_val = torch.tensor(val_df['label'].values, dtype=torch.long)

        X_test = [truncate_sequence_from_ends(seq, self.max_sequence_length) 
                  for seq in test_df['sequence']]
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

        # Create datasets: (sequence, label, conditional_input) tuples
        train_dataset = [(seq, label, np.array([])) for seq, label in zip(X_train, y_train)]
        validation_dataset = [(seq, label, np.array([])) for seq, label in zip(X_val, y_val)]
        test_dataset = [(seq, label, np.array([])) for seq, label in zip(X_test, y_test)]

        return train_dataset, validation_dataset, test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
