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
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_variant_effect_task import (
    BaseGFMSupervisedVariantEffectTask,
)
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant, standardize_sequence


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
        chrom = str(row["chrom"])
        pos1 = int(row["pos"])
        alt = str(row["alt"]).upper()
        label = int(row["label"])
        tissue_id = self.tissue_map.get(str(row.get("tissue", "unknown")), 0)

        pos_0 = pos1 - 1
        ref_seq_raw = pad_sequence_centered_variant(
            chromosome=self.fasta[chrom],
            variant_pos_0based=pos_0,
            max_sequence_length=self.seq_len,
            variant_pos_in_seq=self.center,
        )
        ref_seq = standardize_sequence(ref_seq_raw)
        alt_seq_list = list(ref_seq)
        alt_seq_list[self.center] = alt
        alt_seq = standardize_sequence("".join(alt_seq_list))
        conditional_input = np.array([tissue_id], dtype=np.float32)
        return alt_seq, ref_seq, label, conditional_input


def _resolve_task_mode(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("task_mode", cfg.get("mode", "supervised"))
    if bool(cfg.get("unsupervised", False)) or bool(cfg.get("zero_shot", False)):
        raw = "unsupervised"
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in {"zero_shot", "zeroshot", "unsupervised", "inference_only"}:
        return "unsupervised"
    return "supervised"


def _load_eqtl_dataframe(root_data_dir_path: str, task_name: str) -> pd.DataFrame:
    task_data_dir = os.path.join(root_data_dir_path, task_name)
    os.makedirs(task_data_dir, exist_ok=True)

    expected_filename = "All_Tissues.csv"
    flat_path = os.path.join(task_data_dir, expected_filename)
    nested_path = os.path.join(task_data_dir, "variant_effect_causal_eqtl", expected_filename)
    if os.path.exists(flat_path):
        csv_path = flat_path
    elif os.path.exists(nested_path):
        csv_path = nested_path
    else:
        csv_path = hf_hub_download(
            repo_id="InstaDeepAI/genomics-long-range-benchmark",
            filename="variant_effect_causal_eqtl/All_Tissues.csv",
            repo_type="dataset",
            local_dir=task_data_dir,
        )

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if "chrom" not in df.columns and "chromosome" in df.columns:
        df = df.rename(columns={"chromosome": "chrom"})
    if "pos" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "pos"})
    if "label" not in df.columns and "int_label" in df.columns:
        df["label"] = df["int_label"].astype(int)
    required = {"chrom", "pos", "alt", "label", "tissue"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"LRB Causal eQTL CSV missing required columns: {missing}")
    df["chrom"] = df["chrom"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    df["alt"] = df["alt"].astype(str).str.upper()
    return df


def _split_eqtl_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        split_vals = df["split"].astype(str).str.lower()
        train_df = df[split_vals == "train"].reset_index(drop=True)
        test_df = df[split_vals == "test"].reset_index(drop=True)
    else:
        test_mask = df["chrom"].isin({"chr8", "8"})
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split sizes for LRB causal eQTL: train={len(train_df)}, test={len(test_df)}"
        )
    return train_df, test_df


def _build_tissue_map(df: pd.DataFrame) -> Dict[str, int]:
    unique_tissues = sorted(df["tissue"].astype(str).unique())
    return {t: i for i, t in enumerate(unique_tissues)}


def _get_hg38_fasta(root_data_dir_path: str) -> Fasta:
    genome_dir = os.path.join(root_data_dir_path, "reference_genome")
    os.makedirs(genome_dir, exist_ok=True)
    genome_path = os.path.join(genome_dir, "hg38.fa")
    ensure_reference_genome(genome_path)
    return Fasta(genome_path, one_based_attributes=False)


class LRBCausalEqtlSupervisedTask(BaseGFMSupervisedVariantEffectTask):
    def get_task_name(self) -> str:
        return "lrb_variant_effect_causal_eqtl"

    def _get_num_labels(self) -> int:
        return 2

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        cfg = self.task_config or {}
        inference_only_test_set = bool(cfg.get("inference_only_test_set", False))
        df = _load_eqtl_dataframe(self.root_data_dir_path, self.get_task_name())
        tissue_map = _build_tissue_map(df)
        self.num_tissues = len(tissue_map)
        train_df, test_df = _split_eqtl_df(df)

        if self.max_num_samples is not None:
            train_df = train_df.head(self.max_num_samples)
            test_df = test_df.head(self.max_num_samples)

        fasta = _get_hg38_fasta(self.root_data_dir_path)
        train_ds = None if inference_only_test_set else _LRBCausalEqtlDataset(
            train_df, fasta, self.max_sequence_length, tissue_map
        )
        test_ds = _LRBCausalEqtlDataset(test_df, fasta, self.max_sequence_length, tissue_map)
        return train_ds, None, test_ds

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        max_tissue_id = max(0, int(getattr(self, "num_tissues", 1)) - 1)
        return pd.DataFrame(
            {
                "meta_data_name": ["tissue_id"],
                "data_type": ["integer"],
                "min_value": [0],
                "max_value": [max_tissue_id],
            }
        )


class LRBCausalEqtlZeroShotTask(BaseGFMZeroShotSNVTask):
    """Zero-shot SNV task; metrics are inherited from BaseGFMZeroShotSNVTask (same as OMIM)."""

    def get_task_name(self) -> str:
        return "lrb_variant_effect_causal_eqtl"

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def _get_variant_position_in_sequence(self) -> int:
        return self.max_sequence_length // 2
    
    def _create_test_dataset(self) -> Dataset:
        df = _load_eqtl_dataframe(self.root_data_dir_path, self.get_task_name())
        tissue_map = _build_tissue_map(df)
        self.num_tissues = len(tissue_map)
        _, test_df = _split_eqtl_df(df)
        if self.max_num_samples is not None:
            test_df = test_df.head(self.max_num_samples)
        fasta = _get_hg38_fasta(self.root_data_dir_path)
        return _LRBCausalEqtlDataset(test_df, fasta, self.max_sequence_length, tissue_map)

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        max_tissue_id = max(0, int(getattr(self, "num_tissues", 1)) - 1)
        return pd.DataFrame(
            {
                "meta_data_name": ["tissue_id"],
                "data_type": ["integer"],
                "min_value": [0],
                "max_value": [max_tissue_id],
            }
        )


class LRBCausalEqtlTask:
    """Config-driven dispatcher for supervised vs. unsupervised eQTL task."""

    def __new__(cls, root_data_dir_path: str, task_config: Optional[Dict[str, Any]] = None):
        cfg = task_config or {}
        mode = _resolve_task_mode(cfg)
        impl_cls = LRBCausalEqtlZeroShotTask if mode == "unsupervised" else LRBCausalEqtlSupervisedTask
        return impl_cls(root_data_dir_path=root_data_dir_path, task_config=task_config)
