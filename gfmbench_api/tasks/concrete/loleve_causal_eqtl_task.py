# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/Marks-lab/LOL-EVE-eQTL_benchmark — MIT

# benchmarks/tasks/concrete/loleve_causal_eqtl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download
from datasets import Dataset as HFDataset
from gfmbench_api.tasks.base.base_gfm_zeroshot_general_indel_task import (
    BaseGFMZeroShotGeneralIndelTask,
)

# -------------------------
# Constants
# -------------------------
PIP_LOW = 0.01
PIP_HIGH = 0.99
DEFAULT_SLIPPAGE_THRESHOLD = 100.0
ALLOW_VARIANT_TYPES = ("insertion", "deletion")
SLIPPAGE_WINDOW_SIZE = 20
MIN_REPEAT_LENGTH = 3


def _standardize_sequence(seq: str) -> str:
    seq = (seq or "").upper()
    return "".join([c if c in {"A", "C", "G", "T"} else "N" for c in seq])


def _strip_shared_prefix(ref: str, alt: str) -> tuple[str, str]:
    """Remove identical leading bases so we isolate inserted/deleted bases for repeat matching."""
    ref = (ref or "").upper()
    alt = (alt or "").upper()
    k = 0
    m = min(len(ref), len(alt))
    while k < m and ref[k] == alt[k]:
        k += 1
    return ref[k:], alt[k:]


def analyze_repeats(sequence: str, min_repeat_length: int = MIN_REPEAT_LENGTH) -> Dict[str, Any]:
    """
    Repeat scanning used to compute a "slippage_score" around the variant.
    NOTE: preserves the original notebook's Python quirk: assigning to `i` inside a for-loop has no effect.
    """
    sequence = (sequence or "").upper()
    results: Dict[str, Any] = {"is_slippage_region": False, "slippage_score": 0.0, "repeats": []}

    # 1) Homopolymer runs
    i = 0
    while i < len(sequence):
        j = i
        base = sequence[i]
        while j < len(sequence) and sequence[j] == base:
            j += 1
        run_len = j - i
        if run_len >= min_repeat_length:
            results["repeats"].append({"type": "homopolymer", "unit": base, "count": run_len, "start": i, "end": j - 1})
            results["is_slippage_region"] = True
            results["slippage_score"] += float(run_len ** 2)
        i = j

    # 2) 2-4mer STRs
    for unit_size in range(2, 5):
        for i in range(len(sequence) - unit_size):
            unit = sequence[i : i + unit_size]
            count = 1
            j = i + unit_size
            while j <= len(sequence) - unit_size and sequence[j : j + unit_size] == unit:
                count += 1
                j += unit_size
            if count >= min_repeat_length:
                rtype = {2: "dinucleotide", 3: "trinucleotide", 4: "tetranucleotide"}[unit_size]
                results["repeats"].append({"type": rtype, "unit": unit, "count": count, "start": i, "end": j - 1})
                results["is_slippage_region"] = True
                weight = {2: 0.8, 3: 0.6, 4: 0.5}[unit_size]
                results["slippage_score"] += float((count * unit_size) ** 1.5 * weight)

                # This assignment is intentionally ineffective inside a Python for-loop,
                # but we keep it to match the notebook's behavior.
                i = j - 1

    return results


def get_slippage_info_from_coords(
    wt_seq: str,
    ref: str,
    alt: str,
    pos0: int,  # 0-based index inside wt_seq
    window_size: int = SLIPPAGE_WINDOW_SIZE,
    min_repeat_length: int = MIN_REPEAT_LENGTH,
) -> Dict[str, Any]:
    """Compute slippage_score within a window around the variant and mark repeat match conditions."""
    wt_seq = (wt_seq or "").upper()
    ref = (ref or "").upper()
    alt = (alt or "").upper()

    is_insertion = len(alt) > len(ref)
    is_deletion = len(ref) > len(alt)
    if not (is_insertion or is_deletion):
        return {"slippage_score": 0.0, "is_slippage_region": False}

    ref_suf, alt_suf = _strip_shared_prefix(ref, alt)
    inserted = alt_suf if is_insertion else ""
    deleted = ref_suf if is_deletion else ""

    half = window_size // 2
    start_idx = max(0, pos0 - half)
    end_idx = min(len(wt_seq), pos0 + half)
    region = wt_seq[start_idx:end_idx]

    results = analyze_repeats(region, min_repeat_length=min_repeat_length)
    rel_pos = pos0 - start_idx

    if is_deletion and deleted:
        for rep in results["repeats"]:
            unit = rep["unit"]
            if deleted == unit or deleted.startswith(unit) or deleted.endswith(unit):
                results["is_slippage_region"] = True
            if rep["start"] <= rel_pos <= rep["end"]:
                results["is_slippage_region"] = True

    if is_insertion and inserted:
        for rep in results["repeats"]:
            unit = rep["unit"]
            if inserted == unit or inserted.startswith(unit) or inserted.endswith(unit):
                results["is_slippage_region"] = True
            if rep["start"] <= rel_pos <= rep["end"]:
                results["is_slippage_region"] = True

    return results


def _center_crop(seq: str, center: int, max_len: int) -> str:
    """Deterministic crop around a center position."""
    if max_len is None or max_len <= 0:
        return seq
    if len(seq) <= max_len:
        return seq
    half = max_len // 2
    start = max(0, center - half)
    end = start + max_len
    if end > len(seq):
        end = len(seq)
        start = max(0, end - max_len)
    return seq[start:end]
    


@dataclass(frozen=True)
class _EqtlConfig:
    max_num_samples: Optional[int]
    max_sequence_length: int
    slippage_threshold: float = DEFAULT_SLIPPAGE_THRESHOLD


class _LOL_EVECausalEqtlDataset(Dataset):
    """Returns (variant_sequence, reference_sequence, label, conditional_inputs=None)."""

    def __init__(self, hf_ds: HFDataset, cfg: _EqtlConfig):
        df = hf_ds.to_pandas()

        required = ["pip", "wt_sequence", "var_sequence", "position", "wt_sequence_start", "ref", "alt", "variant_type"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns {missing}. Available: {list(df.columns)}")

        # 1) Keep only indels 
        vt = df["variant_type"].astype(str).str.lower()
        df = df.loc[vt.isin(set(ALLOW_VARIANT_TYPES))].copy().reset_index(drop=True)

        # 2) Symmetric PIP filtering: keep confident extremes only
        pip = df["pip"].to_numpy(dtype=np.float32)
        keep = (pip >= PIP_HIGH) | (pip <= PIP_LOW)
        df = df.loc[keep].copy().reset_index(drop=True)

        pip = df["pip"].to_numpy(dtype=np.float32)
        labels = (pip >= PIP_HIGH).astype(np.int64)

        # 3) Slippage scoring and filtering
        wt = df["wt_sequence"].astype(str).tolist()
        var = df["var_sequence"].astype(str).tolist()
        ref = df["ref"].astype(str).tolist()
        alt = df["alt"].astype(str).tolist()
        pos = df["position"].to_numpy(dtype=np.int64)
        wt_start = df["wt_sequence_start"].to_numpy(dtype=np.int64)

        slip_scores: List[float] = []
        for w, r, a, p, ws in zip(wt, ref, alt, pos, wt_start):
            pos0 = int(p - ws)
            if pos0 < 0 or pos0 >= len(w):
                slip_scores.append(0.0)
                continue
            res = get_slippage_info_from_coords(w, r, a, pos0)
            slip_scores.append(float(res["slippage_score"]))
        slip_scores_arr = np.asarray(slip_scores, dtype=np.float32)

        keep_slip = slip_scores_arr <= float(cfg.slippage_threshold)
        df = df.loc[keep_slip].reset_index(drop=True)
        labels = labels[keep_slip]

        # 4) Standardize + crop to framework max_sequence_length around the variant position
        wt = df["wt_sequence"].astype(str).tolist()
        var = df["var_sequence"].astype(str).tolist()
        pos = df["position"].to_numpy(dtype=np.int64)
        wt_start = df["wt_sequence_start"].to_numpy(dtype=np.int64)

        wt_out: List[str] = []
        var_out: List[str] = []
        for w, v, p, ws in zip(wt, var, pos, wt_start):
            w = _standardize_sequence(w)
            v = _standardize_sequence(v)
            center = int(p - ws)
            wt_out.append(_center_crop(w, center=center, max_len=int(cfg.max_sequence_length)))
            var_out.append(_center_crop(v, center=center, max_len=int(cfg.max_sequence_length)))

        # 5) Deterministic max_num_samples
        if cfg.max_num_samples is not None:
            n = int(cfg.max_num_samples)
            wt_out = wt_out[:n]
            var_out = var_out[:n]
            labels = labels[:n]


        self._wt = wt_out
        self._var = var_out
        self._labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self._labels))

    def __getitem__(self, i: int):
        return self._var[i], self._wt[i], int(self._labels[i]), 0

    @property
    def labels(self) -> np.ndarray:
        return self._labels


class LoleveCausalEqtlTask(BaseGFMZeroShotGeneralIndelTask):
    def get_task_name(self) -> str:
        return "loleve_causal_eqtl"

    def _get_default_max_seq_len(self) -> int:
        # framework should pass max_sequence_length; keep a safe default here
        return 131072

    def _is_snv_only(self) -> bool:
        return False

    def get_conditional_input_meta_data_frame(self):
        return None

    def _create_test_dataset(self) -> Dataset:
        # max_seq = int(self.task_config.get("max_sequence_length", self._get_default_max_seq_len()))
        max_seq = self.max_sequence_length
        # max_n = self.task_config.get("max_num_samples", None)
        max_n = self.max_num_samples
        cfg = _EqtlConfig(max_num_samples=max_n, max_sequence_length=max_seq)

        arrow_path = hf_hub_download(
            repo_id="Marks-lab/LOL-EVE-eQTL_benchmark",
            repo_type="dataset",
            filename="dataset/data-00000-of-00001.arrow",
        )
        hf_ds = HFDataset.from_file(arrow_path)
        return _LOL_EVECausalEqtlDataset(hf_ds, cfg)
