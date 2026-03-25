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
# - https://huggingface.co/datasets/m42-health/variant-benchmark — CC-BY-NC-4.0
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import DatasetDict

from gfmbench_api.tasks.base.base_gfm_supervised_variant_effect_task import BaseGFMSupervisedVariantEffectTask
import numpy as np
from gfmbench_api.utils.fileutils import download_hf_dataset_files
from gfmbench_api.utils.preprocutils import build_forward_centered_seqs

#BioFM based chromosome splits for cross-validation
FOLD_SPLIT = {
    0: ['chr1', 'chr2'],
    1: ['chr3', 'chr4'],
    2: ['chr5', 'chr6'],
    3: ['chr7', 'chr8'],
    4: ['chr9', 'chr10'],
    5: ['chr11', 'chr12'],
    6: ['chr13', 'chr14'],
    7: ['chr15', 'chr16'],
    8: ['chr17', 'chr18'],
    9: ['chr19', 'chr20'],
    10: ['chr21', 'chr22', 'chrX'],
}

def get_fold_split(fold: Optional[int] = None, split_name: str = 'test') -> List[str]:
    if fold is None:
        fold = 0
    if split_name == 'test':
        return FOLD_SPLIT[fold]
    else:
        raise ValueError(f'Unknown split name: {split_name}, should be test')


class VariantBenchmarksMEQTLTask(BaseGFMSupervisedVariantEffectTask):
    """
    Variant sQTL prediction task from m42-health/variant-benchmark dataset -
    Predicting whether a genetic variant influences nearby methylation rates.
    Derived from GRASP which contains 52418 datapoints, 25496 after deduplication, 
    For negative controls, twice as much variants from variants with more than 5% MAF were randomly sampled from Gnomad
    
    The dataset contains reference and variation sequences labeled as altering nearby methylation or not,
    in the following annotated format:
    alt_left - Alternate allele, left strand: SNP is at alt_let[-1]
    alt_right - Alternate allele, right strand (reverse complement): RC-SNP is at alt_right[-1]
    ref_left - Reference allele, left strand: SNP is at alt_let[-1]
    ref_right - Reference allele, right strand (reverse complement): RC-SNP is at alt_right[-1]
    Notes:
    - Original sequences are whitespace separated.
    - Transcript, Exon, and CDS boundaries tags are added to left sequences, 
        e.g.,  [START_TRANSCRIPT] [START_EXON] [START_CDS]
    - SNPs are saved using the following conversion map:
        VARIANT_MAP = {"A": "Â", "C": "Ĉ", "G": "Ĝ", "T": "Ṱ", "N": "N"}
    (See https://github.com/m42-health/biofm-eval/blob/main/biofm_eval/data_utils.py, annotate_snp_record)
         https://github.com/m42-health/biofm-eval/blob/main/biofm_eval/annotators/base.py

    For reproduction, Linear probing is done originally with LogisticRegression(max_iter=5000) on embeddings generated on all four annotated sequences.
    (See https://github.com/m42-health/biofm-eval/blob/main/biofm_eval/embedder.py, linear_probing)

    In order to support the general "ref_seq", "alt_seq", "label" format, this task optionally extracts forward-strand, SNP-centered var and ref sequences
    """
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None):
        # HuggingFace dataset source
        self.hf_repo_id = "m42-health/variant-benchmark"
        self.hf_subfolder = "meqtl"
        
        self.test_fold = 10 # Forcing the last fold to be the test set
        
        super().__init__(root_data_dir_path, task_config)

    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length (1024bp)."""
        return 1024 # Reported paper's default
    
    def _get_num_labels(self):
        """Return 2 (binary: affects nearby methylation vs. not)."""
        return 2

    def get_task_name(self):
        """Return task name (identical to data directory name)."""
        return "var_bench_meqtl"
    
    def _create_datasets(self):
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())

        # Download data if not exists
        logging.info(f"Downloading {self.get_task_name()} from HuggingFace...")
        download_hf_dataset_files(
            repo_id=self.hf_repo_id,
            subfolder=self.hf_subfolder,
            splits=["train"],
            local_dir=data_dir,
            concat_tasks=False
        )
        logging.info(f"Data saved to: {data_dir}")

        # Load parquets
        whole_ds = DatasetDict.load_from_disk(data_dir, keep_in_memory=True)

        train_ds = whole_ds['train'].filter(lambda x: x['chrom'] not in get_fold_split(self.test_fold, 'test'), keep_in_memory=True)
        test_ds = whole_ds['train'].filter(lambda x: x['chrom'] in get_fold_split(self.test_fold, 'test'), keep_in_memory=True)

        train_df = train_ds.to_pandas()
        test_df = test_ds.to_pandas()
        
        # Limit samples if max_num_samples is specified
        if self.max_num_samples is not None:
            train_df = train_df.head(min(self.max_num_samples, len(train_df)))
            test_df = test_df.head(min(self.max_num_samples, len(test_df)))


        # Create dataset: (variant_sequence, reference_sequence, label, conditional_input) tuples
        # conditional_input is None for this task (no metadata)
        X_train = build_forward_centered_seqs(train_df)
        y_train = torch.tensor(train_df['label'].values, dtype=torch.long)

        train_dataset = [(var_seq, ref_seq, label, np.array([])) 
                            for var_seq, ref_seq, label in 
                            zip(X_train['variant_sequence'], X_train['reference_sequence'], y_train)]
        
        X_test = build_forward_centered_seqs(test_df)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)
        
        test_dataset = [(var_seq, ref_seq, label, np.array([])) 
                            for var_seq, ref_seq, label in 
                            zip(X_test['variant_sequence'], X_test['reference_sequence'], y_test)]         
        

        return train_dataset, None, test_dataset
    
    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
    
