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


class _PathogenicClinvarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fasta: Fasta, seq_len: int):
        self.fasta = fasta
        self.seq_len = seq_len
        self.center = seq_len // 2

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df[df["alt"].astype(str).str.len() == 1]
        df["alt"] = df["alt"].astype(str).str.upper()
        df = df[df["alt"].isin(["A", "C", "G", "T"])]

        valid_indices = []
        for idx in df.index:
            try:
                chrom_raw = str(df.at[idx, "chrom"])
                chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
                pos = int(df.at[idx, "pos"])
                if chrom not in fasta.keys():
                    continue
                chrom_len = len(fasta[chrom])
                start = (pos - 1) - self.center
                end = start + self.seq_len
                if start >= 0 and end <= chrom_len:
                    valid_indices.append(idx)
            except Exception:
                continue

        self.df = df.loc[valid_indices].reset_index(drop=True)
        logging.info("ClinVar LRB: kept %d valid samples after filtering.", len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        chrom_raw = str(row["chrom"])
        chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
        pos1 = int(row["pos"])
        alt = str(row["alt"]).upper()
        label = int(row["label"])

        pos_0 = pos1 - 1
        ref_seq_raw = pad_sequence_centered_variant(
            chromosome=self.fasta[chrom],
            variant_pos_0based=pos_0,
            max_sequence_length=self.seq_len,
            variant_pos_in_seq=self.center,
        )
        ref_seq = standardize_sequence(ref_seq_raw)
        var_seq_list = list(ref_seq)
        var_seq_list[self.center] = alt
        var_seq = standardize_sequence("".join(var_seq_list))
        return var_seq, ref_seq, label, np.array([], dtype=np.float32)


def _resolve_task_mode(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("task_mode", cfg.get("mode", "supervised"))
    if bool(cfg.get("unsupervised", False)) or bool(cfg.get("zero_shot", False)):
        raw = "unsupervised"
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in {"zero_shot", "zeroshot", "unsupervised", "inference_only"}:
        return "unsupervised"
    return "supervised"


def _coerce_binary_label_column(df: pd.DataFrame) -> pd.DataFrame:
    if "int_label" in df.columns:
        df["label"] = df["int_label"].astype(int)
        return df
    if "label" not in df.columns:
        raise ValueError("ClinVar CSV must contain 'label' or 'INT_LABEL' column.")
    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
        return df

    label_map = {
        "pathogenic": 1,
        "likely_pathogenic": 1,
        "benign": 0,
        "likely_benign": 0,
    }
    mapped = df["label"].astype(str).str.strip().str.lower().map(label_map)
    if mapped.isna().any():
        unknown = sorted(df.loc[mapped.isna(), "label"].astype(str).unique())
        raise ValueError(
            "ClinVar label column contains unmapped non-numeric values: "
            f"{unknown[:5]}"
        )
    df["label"] = mapped.astype(int)
    return df


def _load_clinvar_dataframe(
    root_data_dir_path: str, task_name: str, variants_path: Optional[str]
) -> pd.DataFrame:
    task_data_dir = os.path.join(root_data_dir_path, task_name)
    os.makedirs(task_data_dir, exist_ok=True)

    if not variants_path:
        candidate_local_files = [
            os.path.join(task_data_dir, "vep_pathogenic_coding.csv"),
            os.path.join(task_data_dir, "variant_effect_pathogenic", "vep_pathogenic_coding.csv"),
            os.path.join(task_data_dir, "vep_pathogenic_clinvar.csv"),
            os.path.join(task_data_dir, "vep_pathogenic_coding_clinvar.csv"),
            os.path.join(task_data_dir, "variant_effect_pathogenic", "vep_pathogenic_clinvar.csv"),
            os.path.join(task_data_dir, "variant_effect_pathogenic", "vep_pathogenic_coding_clinvar.csv"),
        ]
        variants_path = next((p for p in candidate_local_files if os.path.exists(p)), None)

        if variants_path is None:
            hf_candidates = [
                "variant_effect_pathogenic/vep_pathogenic_coding.csv",
                "variant_effect_pathogenic/vep_pathogenic_clinvar.csv",
                "variant_effect_pathogenic/vep_pathogenic_coding_clinvar.csv",
            ]
            last_error = None
            for hf_file in hf_candidates:
                try:
                    variants_path = hf_hub_download(
                        repo_id="InstaDeepAI/genomics-long-range-benchmark",
                        filename=hf_file,
                        repo_type="dataset",
                        local_dir=task_data_dir,
                    )
                    break
                except Exception as e:
                    last_error = e
                    continue
            if variants_path is None:
                raise RuntimeError(
                    "Failed to locate/download LRB ClinVar variants file. "
                    "Provide task_config['variants_path'] pointing to the ClinVar CSV. "
                    f"Last download error: {last_error}"
                )

    logging.info("[Task] Using LRB ClinVar variants file: %s", variants_path)
    df = pd.read_csv(variants_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if "chrom" not in df.columns and "chromosome" in df.columns:
        df = df.rename(columns={"chromosome": "chrom"})
    if "pos" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "pos"})
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower()
    return _coerce_binary_label_column(df)


def _split_clinvar_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        if "chrom" not in df.columns:
            raise ValueError(
                "Could not determine train/test split: no split column and no chrom column in dataset."
            )
        chrom_vals = df["chrom"].astype(str)
        test_mask = chrom_vals.isin({"chr8", "8"})
        test_df = df[test_mask].copy()
        train_df = df[~test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split sizes for ClinVar LRB task: train={len(train_df)}, test={len(test_df)}"
        )
    return train_df, test_df


def _get_hg38_fasta(root_data_dir_path: str) -> Fasta:
    reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
    if not os.path.exists(reference_genome_path):
        logging.info("Reference genome not found. Downloading hg38.fa...")
        ensure_reference_genome(reference_genome_path)
    return Fasta(str(reference_genome_path), one_based_attributes=False)


class LrbVariantEffectPathogenicClinvarSupervisedTask(BaseGFMSupervisedVariantEffectTask):
    def get_task_name(self) -> str:
        return "lrb_variant_effect_pathogenic_clinvar"

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def _get_num_labels(self) -> int:
        return 2

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        cfg = self.task_config or {}
        inference_only_test_set = bool(cfg.get("inference_only_test_set", False))
        df = _load_clinvar_dataframe(
            root_data_dir_path=self.root_data_dir_path,
            task_name=self.get_task_name(),
            variants_path=cfg.get("variants_path"),
        )
        train_df, test_df = _split_clinvar_df(df)
        if self.max_num_samples is not None:
            train_df = train_df.head(min(self.max_num_samples, len(train_df)))
            test_df = test_df.head(min(self.max_num_samples, len(test_df)))

        fasta = _get_hg38_fasta(self.root_data_dir_path)
        train_dataset = None if inference_only_test_set else _PathogenicClinvarDataset(
            train_df, fasta, self.max_sequence_length
        )
        test_dataset = _PathogenicClinvarDataset(test_df, fasta, self.max_sequence_length)
        return train_dataset, None, test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        return None


class LrbVariantEffectPathogenicClinvarZeroShotTask(BaseGFMZeroShotSNVTask):
    """Zero-shot SNV task; metrics are inherited from BaseGFMZeroShotSNVTask (same as OMIM)."""

    def get_task_name(self) -> str:
        return "lrb_variant_effect_pathogenic_clinvar"

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def _get_variant_position_in_sequence(self) -> int:
        return self.max_sequence_length // 2

    def _create_test_dataset(self) -> Dataset:
        cfg = self.task_config or {}
        df = _load_clinvar_dataframe(
            root_data_dir_path=self.root_data_dir_path,
            task_name=self.get_task_name(),
            variants_path=cfg.get("variants_path"),
        )
        _, test_df = _split_clinvar_df(df)
        if self.max_num_samples is not None:
            test_df = test_df.head(min(self.max_num_samples, len(test_df)))
        fasta = _get_hg38_fasta(self.root_data_dir_path)
        return _PathogenicClinvarDataset(test_df, fasta, self.max_sequence_length)

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        return None


class LrbVariantEffectPathogenicClinvarTask:
    """Config-driven dispatcher for supervised vs. unsupervised ClinVar task."""

    def __new__(cls, root_data_dir_path: str, task_config: Optional[Dict[str, Any]] = None):
        cfg = task_config or {}
        mode = _resolve_task_mode(cfg)
        impl_cls = (
            LrbVariantEffectPathogenicClinvarZeroShotTask
            if mode == "unsupervised"
            else LrbVariantEffectPathogenicClinvarSupervisedTask
        )
        return impl_cls(root_data_dir_path=root_data_dir_path, task_config=task_config)
