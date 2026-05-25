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

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask


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


def _reverse_complement_dna(seq: str) -> str:
    table = str.maketrans("ATCGNatcgn", "TAGCNtagcn")
    return seq.translate(table)[::-1]


def _label_to_int(label: str) -> int:
    return 1 if str(label).lower() == "positive" else 0


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


def _mask_intervening_regions(
    seq: str,
    sequence_start: int,
    chrom: str,
    mask_tabix: Any,
    query_start: int,
    query_end: int,
) -> str:
    result = seq
    overlaps = _safe_tabix_query(mask_tabix, chrom, query_start, query_end)
    for overlap in overlaps:
        o_start = int(overlap[1])
        o_end = int(overlap[2])
        rel_start = o_start - sequence_start
        rel_end = o_end - sequence_start
        if rel_start < 0 or rel_end <= 0:
            continue
        if rel_start >= len(result):
            continue
        left = max(0, rel_start)
        right = min(len(result), rel_end)
        if right > left:
            result = result[:left] + ("N" * (right - left)) + result[right:]
    return result


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


def _center_window_with_padding(seq: str, center_pos: int, output_len: int) -> str:
    half = output_len // 2
    start = center_pos - half
    end = start + output_len
    pad_left = max(0, -start)
    pad_right = max(0, end - len(seq))
    start = max(0, start)
    end = min(len(seq), end)
    centered = ("N" * pad_left) + seq[start:end] + ("N" * pad_right)
    if len(centered) != output_len:
        centered = centered[:output_len].ljust(output_len, "N")
    return centered


def _resolve_eqtl_root(root_data_dir_path: str, task_name: str, cfg: Dict[str, Any]) -> str:
    explicit = cfg.get("eqtl_data_dir") or cfg.get("data_dir")
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(root_data_dir_path, explicit)

    candidates = [
        os.path.join(root_data_dir_path, task_name),
        os.path.join(root_data_dir_path, "dnalongbench_eqtl"),
        os.path.join(root_data_dir_path, "eQTL"),
        os.path.join(root_data_dir_path, "eqtl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(root_data_dir_path, "eQTL")


class DNALongBenchEqtlTask(BaseGFMZeroShotSNVTask):
    """DNALongBench eQTL as a zero-shot SNV task."""

    def __init__(self, root_data_dir_path: str, task_config: Optional[Dict[str, Any]] = None):
        self._cfg = task_config or {}
        self.eqtl_subset = self._cfg.get("eqtl_subset", "Whole_Blood")
        self.eqtl_root = _resolve_eqtl_root(root_data_dir_path, self.get_task_name(), self._cfg)
        self.num_tissues = 1
        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        # keep exact requested identifier
        return "dnaongbench_eqtl"

    def _get_default_max_seq_len(self) -> int:
        # DNALongBench eQTL defaults are very long, but framework truncates to model max.
        return 450000

    def _get_variant_position_in_sequence(self) -> int:
        return self.max_sequence_length // 2

    def is_diploid_task(self) -> bool:
        return False

    def _create_test_dataset(self):
        config_file = self._cfg.get(
            "config_file",
            os.path.join(self.eqtl_root, "config", f"gtex_hg38.{self.eqtl_subset}.config"),
        )
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"DNALongBench eQTL config not found: {config_file}. "
                "Set task_config['config_file'] or task_config['eqtl_data_dir']."
            )

        cfg = _parse_config(config_file)
        if "genome_fa" not in cfg:
            raise ValueError("Missing `genome_fa` in DNALongBench eQTL config.")

        eqtl_file_key = "eQTL_file" if "eQTL_file" in cfg else "eqtl_file"
        tabix_file_key = "eQTL_tabix_file" if "eQTL_tabix_file" in cfg else "eqtl_tabix_file"
        if eqtl_file_key not in cfg or tabix_file_key not in cfg:
            raise ValueError("Config must contain eQTL_file and eQTL_tabix_file keys.")

        fasta_reader = _FastaStringExtractor(os.path.join(self.eqtl_root, cfg["genome_fa"]))
        df = pd.read_csv(os.path.join(self.eqtl_root, cfg[eqtl_file_key]), sep="\t", header=0)
        mask_tabix = tabix.open(os.path.join(self.eqtl_root, cfg[tabix_file_key]))

        tissue_col = "tissue" if "tissue" in df.columns else ("gene_name" if "gene_name" in df.columns else None)
        unique_tissues = sorted(df[tissue_col].astype(str).unique()) if tissue_col is not None else ["unknown"]
        tissue_map = {t: i for i, t in enumerate(unique_tissues)}
        self.num_tissues = max(1, len(tissue_map))

        tss_up = int(cfg["tss_flank_upstream"])
        tss_down = int(cfg["tss_flank_downstream"])
        region_up = int(cfg["region_flank_upstream"])
        region_down = int(cfg["region_flank_downstream"])

        records = []
        for _, row in df.iterrows():
            if str(row.get("subset", "")).lower() != "test":
                continue

            gene_chrom = row["gene_chrom"]
            region_chrom = row["region_chrom"]
            if gene_chrom != region_chrom:
                continue

            allele1 = str(row["allele1"]).upper()
            allele2 = str(row["allele2"]).upper()
            variant_start = int(row["region_start"])
            variant_end = int(row["region_end"])
            if len(allele1) != 1 or len(allele2) != 1:
                continue
            if variant_end - variant_start != 1:
                continue

            if row["gene_strand"] == "+":
                tss_start = int(row["gene_start"])
            elif row["gene_strand"] == "-":
                tss_start = int(row["gene_end"]) - 1
            else:
                continue
            tss_end = tss_start + 1

            tss_start -= tss_up
            tss_end += tss_down
            region_start = variant_start - region_up
            region_end = variant_end + region_down

            sequence_start = min(tss_start, region_start)
            sequence_end = max(tss_end, region_end)
            seq_ref = fasta_reader.extract(region_chrom, sequence_start, sequence_end)

            rel_var_start = variant_start - sequence_start
            rel_var_end = variant_end - sequence_start
            if rel_var_end > len(seq_ref) or rel_var_start < 0:
                continue
            if seq_ref[rel_var_start:rel_var_end].upper() != allele1:
                continue

            distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
            if distance > 0:
                if tss_start > region_end:
                    query_start, query_end = region_end, tss_start
                else:
                    query_start, query_end = tss_end, region_start
                seq_ref = _mask_intervening_regions(
                    seq_ref,
                    sequence_start,
                    region_chrom,
                    mask_tabix,
                    query_start,
                    query_end,
                )

            seq_alt = seq_ref[:rel_var_start] + allele2 + seq_ref[rel_var_end:]
            center_pos = rel_var_start

            if int(row["gene_start"]) > region_end:
                seq_ref = _reverse_complement_dna(seq_ref)
                seq_alt = _reverse_complement_dna(seq_alt)
                center_pos = len(seq_ref) - 1 - center_pos

            seq_ref_centered = _center_window_with_padding(seq_ref, center_pos, self.max_sequence_length)
            seq_alt_centered = _center_window_with_padding(seq_alt, center_pos, self.max_sequence_length)

            label = _label_to_int(str(row["target"]))
            tissue_name = str(row.get(tissue_col, "unknown")) if tissue_col is not None else "unknown"
            conditional_input = np.array([tissue_map.get(tissue_name, 0)], dtype=np.float32)
            records.append((seq_alt_centered, seq_ref_centered, label, conditional_input))

            if self.max_num_samples is not None and len(records) >= self.max_num_samples:
                break

        if len(records) == 0:
            raise ValueError("No valid DNALongBench eQTL test samples were created.")
        return records

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


if __name__ == "__main__":
    task = DNALongBenchEqtlTask(root_data_dir_path="/mnt/lustre/users/elayd/benchmark_data")
    task._create_test_dataset()
    print(task.test_dataset)
    print(task.get_conditional_input_meta_data_frame())