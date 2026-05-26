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
#
# This module does not embed third-party data download URLs.
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyfaidx
import tabix

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import (
    BaseGFMSupervisedSingleSeqTask,
)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_config(config_file: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    with open(config_file, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            row = line.split()
            if len(row) < 3:
                continue
            key, value, data_type = row[0], row[1], row[2]
            if data_type == "int":
                parsed: Any = int(value)
            elif data_type == "float":
                parsed = float(value)
            elif data_type == "bool":
                parsed = _parse_bool(value)
            elif data_type == "list":
                parsed = [v for v in value.split(",")]
            else:
                parsed = value
            cfg[key] = parsed
    return cfg


def _safe_tabix_query(
    tabix_file: Any,
    chrom: str,
    start: int,
    end: int,
) -> Iterable[Sequence[str]]:
    if end <= start:
        return []
    try:
        return tabix_file.query(chrom, max(0, int(start)), max(0, int(end)))
    except Exception:
        return []


def _reverse_complement_dna(seq: str) -> str:
    table = str.maketrans("ATCGNatcgn", "TAGCNtagcn")
    return seq.translate(table)[::-1]


class _FastaStringExtractor:
    def __init__(self, fasta_file: str):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, chrom: str, start: int, end: int) -> str:
        chromosome_length = self._chromosome_sizes[chrom]
        trimmed_start = max(start, 0)
        trimmed_end = min(end, chromosome_length)
        sequence = str(self.fasta.get_seq(chrom, trimmed_start + 1, trimmed_end).seq).upper()
        pad_upstream = "N" * max(-start, 0)
        pad_downstream = "N" * max(end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream


def _label_to_int(label: str) -> int:
    return 1 if str(label).strip().lower() == "positive" else 0


def _parse_epi_subset(
    df: pd.DataFrame,
    subset_name: str,
    fasta_reader: _FastaStringExtractor,
    mask_tabix: Any,
    seq_len_cutoff: int,
    tss_flank_upstream: int,
    tss_flank_downstream: int,
    region_flank_upstream: int,
    region_flank_downstream: int,
    max_samples: Optional[int],
) -> List[tuple]:
    records: List[tuple] = []
    for _, row in df.iterrows():
        if str(row.get("subset", "")).lower() != subset_name:
            continue

        gene_chrom = str(row["gene_chrom"])
        region_chrom = str(row["region_chrom"])
        if gene_chrom != region_chrom:
            continue

        gene_start = int(row["gene_start"])
        gene_end = int(row["gene_end"])
        region_start = int(row["region_start"])
        region_end = int(row["region_end"])

        strand = str(row["gene_strand"])
        if strand == "+":
            tss_start = gene_start
        elif strand == "-":
            tss_start = gene_end - 1
        else:
            continue
        tss_end = tss_start + 1

        tss_start = tss_start - tss_flank_upstream
        tss_end = tss_end + tss_flank_downstream
        region_start = region_start - region_flank_upstream
        region_end = region_end + region_flank_downstream

        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        if distance > seq_len_cutoff:
            continue

        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        sequence = fasta_reader.extract(region_chrom, sequence_start, sequence_end)

        if distance > 0:
            if tss_start > region_end:
                query_start, query_end = region_end, tss_start
            else:
                query_start, query_end = tss_end, region_start
            overlaps = _safe_tabix_query(mask_tabix, region_chrom, query_start, query_end)
            for overlap in overlaps:
                overlap_start = int(overlap[1])
                overlap_end = int(overlap[2])
                rel_start = overlap_start - sequence_start
                rel_end = overlap_end - sequence_start
                left = max(0, rel_start)
                right = min(len(sequence), rel_end)
                if right > left:
                    sequence = sequence[:left] + ("N" * (right - left)) + sequence[right:]

        if gene_start > region_end:
            sequence = _reverse_complement_dna(sequence)

        if len(sequence) <= seq_len_cutoff:
            sequence = sequence + ("N" * (seq_len_cutoff - len(sequence)))
        else:
            sequence = sequence[:seq_len_cutoff]

        label = _label_to_int(str(row["target"]))
        records.append((sequence, label, np.array([], dtype=np.float32)))

        if max_samples is not None and len(records) >= max_samples:
            break
    return records


def _resolve_enhancer_target_gene_root(
    root_data_dir_path: str, task_name: str, cfg: Dict[str, Any]
) -> str:
    explicit = cfg.get("enhancer_target_gene_data_dir") or cfg.get("data_dir")
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(root_data_dir_path, explicit)

    candidates = [
        os.path.join(root_data_dir_path, task_name),
        os.path.join(root_data_dir_path, "enhancer_target_gene"),
        os.path.join(root_data_dir_path, "enhancer_target_gene_prediction"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(root_data_dir_path, "enhancer_target_gene")


class DNALongBenchEnhancerTargetGeneTask(BaseGFMSupervisedSingleSeqTask):
    """DNALongBench enhancer-target-gene prediction as supervised sequence classification."""

    def __init__(self, root_data_dir_path: str, task_config: Optional[Dict[str, Any]] = None):
        self._cfg = task_config or {}
        self.etg_root = _resolve_enhancer_target_gene_root(
            root_data_dir_path, self.get_task_name(), self._cfg
        )
        super().__init__(root_data_dir_path=root_data_dir_path, task_config=task_config)

    def get_task_name(self) -> str:
        return "dnaongbench_enhancer_target_gene"

    def _get_default_max_seq_len(self) -> int:
        return 450000

    def _get_num_labels(self) -> int:
        return 2

    def _create_datasets(self):
        config_file = self._cfg.get(
            "config_file",
            os.path.join(self.etg_root, "config", "CRISPRi_EPI_K562_hg19.config"),
        )
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"DNALongBench enhancer-target-gene config not found: {config_file}. "
                "Set task_config['config_file'] or task_config['enhancer_target_gene_data_dir']."
            )

        cfg = _parse_config(config_file)
        required_keys = [
            "genome_fa",
            "EPI_file",
            "enhancer_tabix_file",
            "seq_len_cutoff",
            "tss_flank_upstream",
            "tss_flank_downstream",
            "region_flank_upstream",
            "region_flank_downstream",
        ]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"Enhancer-target-gene config missing required keys: {missing}")

        fasta_reader = _FastaStringExtractor(os.path.join(self.etg_root, cfg["genome_fa"]))
        mask_tabix = tabix.open(os.path.join(self.etg_root, cfg["enhancer_tabix_file"]))
        epi_df = pd.read_csv(os.path.join(self.etg_root, cfg["EPI_file"]), sep="\t", header=0)

        seq_len_cutoff = int(cfg["seq_len_cutoff"])
        seq_len_cutoff = min(seq_len_cutoff, self.max_sequence_length)

        train_dataset = _parse_epi_subset(
            df=epi_df,
            subset_name="train",
            fasta_reader=fasta_reader,
            mask_tabix=mask_tabix,
            seq_len_cutoff=seq_len_cutoff,
            tss_flank_upstream=int(cfg["tss_flank_upstream"]),
            tss_flank_downstream=int(cfg["tss_flank_downstream"]),
            region_flank_upstream=int(cfg["region_flank_upstream"]),
            region_flank_downstream=int(cfg["region_flank_downstream"]),
            max_samples=self.max_num_samples,
        )
        validation_dataset = _parse_epi_subset(
            df=epi_df,
            subset_name="valid",
            fasta_reader=fasta_reader,
            mask_tabix=mask_tabix,
            seq_len_cutoff=seq_len_cutoff,
            tss_flank_upstream=int(cfg["tss_flank_upstream"]),
            tss_flank_downstream=int(cfg["tss_flank_downstream"]),
            region_flank_upstream=int(cfg["region_flank_upstream"]),
            region_flank_downstream=int(cfg["region_flank_downstream"]),
            max_samples=self.max_num_samples,
        )
        test_dataset = _parse_epi_subset(
            df=epi_df,
            subset_name="test",
            fasta_reader=fasta_reader,
            mask_tabix=mask_tabix,
            seq_len_cutoff=seq_len_cutoff,
            tss_flank_upstream=int(cfg["tss_flank_upstream"]),
            tss_flank_downstream=int(cfg["tss_flank_downstream"]),
            region_flank_upstream=int(cfg["region_flank_upstream"]),
            region_flank_downstream=int(cfg["region_flank_downstream"]),
            max_samples=self.max_num_samples,
        )

        if len(train_dataset) == 0:
            raise ValueError("No valid DNALongBench enhancer-target-gene train samples were created.")
        if len(validation_dataset) == 0:
            raise ValueError("No valid DNALongBench enhancer-target-gene validation samples were created.")
        if len(test_dataset) == 0:
            raise ValueError("No valid DNALongBench enhancer-target-gene test samples were created.")

        return train_dataset, validation_dataset, test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        return None
