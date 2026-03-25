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
# - https://huggingface.co/datasets/leannmlindsey/GUE — MIT
import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import BaseGFMSupervisedSingleSeqTask
import numpy as np
from gfmbench_api.utils.fileutils import download_hf_dataset_files
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends

class GueTranscriptionFactorTask(BaseGFMSupervisedSingleSeqTask):
    """GUE Transcription Factor prediction task (binary classification)."""
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None):
        # HuggingFace dataset source
        self.hf_repo_id = "leannmlindsey/GUE"
        self.hf_subfolder = ['human_tf_0', 'human_tf_1', 'human_tf_2', 'human_tf_3', 'human_tf_4']
        
        super().__init__(root_data_dir_path, task_config)

    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length (100bp)."""
        return 100
    
    def _get_num_labels(self):
        """Return 2 (binary: TF binding site vs non-binding site)."""
        return 2

    def get_task_name(self):
        """Return task name (identical to data directory name)."""
        return "gue_transcription_factor"
    
    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        """Create train, validation, and test datasets from CSV files."""
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "dev.csv")
        test_path = os.path.join(data_dir, "test.csv")

        # Download data if not exists
        if not all(os.path.exists(p) for p in [train_path, test_path]):
            logging.info(f"Downloading {self.get_task_name()} from HuggingFace...")
            download_hf_dataset_files(
                repo_id=self.hf_repo_id,
                subfolder=self.hf_subfolder,
                splits=["train", "test", "dev"],
                local_dir=data_dir,
                concat_tasks=True
            )
            logging.info(f"Data saved to: {data_dir}")

        # Load CSVs
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Limit samples if max_num_samples is specified
        if self.max_num_samples is not None:
            train_df = train_df.head(min(self.max_num_samples, len(train_df)))
            val_df = val_df.head(min(self.max_num_samples, len(val_df)))
            test_df = test_df.head(min(self.max_num_samples, len(test_df)))

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
    