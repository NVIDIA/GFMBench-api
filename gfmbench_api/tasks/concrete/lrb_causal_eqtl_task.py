# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark — CC-BY-NC-4.0
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pyfaidx import Fasta
from huggingface_hub import hf_hub_download
from typing import Any, Tuple, Optional, Dict, List

# Framework Imports
from gfmbench_api.tasks.base.base_gfm_supervised_variant_effect_task import BaseGFMSupervisedVariantEffectTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import standardize_sequence, pad_sequence_centered_variant


class _LRBCausalEqtlDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fasta: Fasta, seq_len: int, tissue_map: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.fasta = fasta
        self.seq_len = seq_len
        self.tissue_map = tissue_map
        self.center = seq_len // 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        chrom, pos1, ref, alt = str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"])
        label = int(row["label"])
        tissue_id = self.tissue_map.get(row.get("tissue", "Unknown"), 0)

        # Extract Context using padding function (handles chromosome boundaries)
        pos_0 = pos1 - 1
        seq_str = pad_sequence_centered_variant(
            chromosome=self.fasta[chrom],
            variant_pos_0based=pos_0,
            max_sequence_length=self.seq_len,
            variant_pos_in_seq=self.center
        )
        seq_list = list(seq_str.upper())
        
        # Inject Alleles
        if 0 <= self.center < len(seq_list):
            seq_list[self.center] = ref 
            ref_seq = "".join(seq_list)
            seq_list[self.center] = alt
            alt_seq = "".join(seq_list)
        else:
            ref_seq, alt_seq = seq_str, seq_str # Fallback

        # conditional_input is a vector of shape (num_metadata,) - here (1,) for tissue_id
        # When batched, becomes (batch_size, 1)
        conditional_input = np.array([tissue_id], dtype=np.float32)
        return standardize_sequence(alt_seq), standardize_sequence(ref_seq), label, conditional_input



# 2. Main Task Class
class LRBCausalEqtlTask(BaseGFMSupervisedVariantEffectTask):
    """
    LRB Causal eQTL Task.
    Includes Data Loading and Twin-Tower Architecture Integration.
    """

    def get_task_name(self) -> str:
        return "lrb_variant_effect_causal_eqtl"

    def _get_num_labels(self) -> int:
        return 2

    def _get_default_max_seq_len(self) -> int:
        return 1048576 

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        task_data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        genome_dir = os.path.join(self.root_data_dir_path, "reference_genome")
        os.makedirs(task_data_dir, exist_ok=True)
        os.makedirs(genome_dir, exist_ok=True)

        # 1. Download/Load Data (Matches framework style)
        local_csv_path = None
        expected_filename = "All_Tissues.csv"
        
        # Check local defaults first (flat structure)
        flat_path = os.path.join(task_data_dir, expected_filename)
        # Check nested structure (if downloaded previously by HF)
        nested_path = os.path.join(task_data_dir, "variant_effect_causal_eqtl", expected_filename)

        if os.path.exists(flat_path):
            local_csv_path = flat_path
        elif os.path.exists(nested_path):
            local_csv_path = nested_path
        else:
            print(f"[Task] Data not found. Downloading from InstaDeepAI/genomics-long-range-benchmark...")
            try:
                # Use return value from hf_hub_download to get exact path
                downloaded_path = hf_hub_download(
                    repo_id="InstaDeepAI/genomics-long-range-benchmark",
                    filename="variant_effect_causal_eqtl/All_Tissues.csv", 
                    repo_type="dataset",
                    local_dir=task_data_dir
                )
                local_csv_path = downloaded_path
            except Exception as e:
                raise RuntimeError(f"Failed to download LRB Causal eQTL data: {e}")

        print(f"[Task] Loading: {local_csv_path}")
        df = pd.read_csv(local_csv_path)
        
        # 2. Preprocessing
        df.columns = [c.lower() for c in df.columns]
        df["chrom"] = df["chrom"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
        
        # 3. Tissue Mapping
        unique_tissues = sorted(df["tissue"].unique().astype(str))
        self.tissue_map = {t: i for i, t in enumerate(unique_tissues)}
        self.num_tissues = len(unique_tissues)
        print(f"[Task] Found {self.num_tissues} unique tissues.")

        # 4. Splits (Train: All except 9,10 | Test: 9,10)
        test_mask = df["chrom"].isin(["chr9", "chr10"])
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        if self.max_num_samples is not None:
            train_df = train_df.head(self.max_num_samples)
            test_df = test_df.head(self.max_num_samples)
        
        # 5. Genome
        genome_path = os.path.join(genome_dir, "hg38.fa")
        ensure_reference_genome(genome_path) 
        fasta = Fasta(genome_path, one_based_attributes=False)

        # 6. Datasets
        seq_len = self.max_sequence_length
        train_ds = _LRBCausalEqtlDataset(train_df, fasta, seq_len, self.tissue_map)
        test_ds = _LRBCausalEqtlDataset(test_df, fasta, seq_len, self.tissue_map)
        
        # Note: returning None for validation set here. 
        # The split is dynamic based on seed in fine_tune() method
        return train_ds, None, test_ds

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """
        Return metadata schema for tissue conditional input.
        
        This task has tissue_id as a conditional input, representing the tissue type
        for the eQTL variant effect prediction.
        
        Returns:
            DataFrame with metadata schema for tissue_id
        """
        return pd.DataFrame({
            'meta_data_name': ['tissue_id'],
            'data_type': ['integer'],
            'min_value': [0],
            'max_value': [self.num_tissues - 1]
        })