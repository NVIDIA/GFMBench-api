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
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz — LicenseRef-UCSC-Genome-Browser
"""Local loaders for the InstaDeep genomics-long-range-benchmark tasks.

Newer ``datasets`` releases (>=4.0) dropped support for dataset *loading
scripts*, so ``load_dataset("InstaDeepAI/genomics-long-range-benchmark", ...)``
fails with "Dataset scripts are no longer supported". These helpers instead
download the raw label files with ``hf_hub_download`` and reproduce the upstream
builder's sequence extraction (``pad_sequence``) exactly, so sequences and
labels match the published dataset.

Each ``build_*`` function returns lists of ``(sequence, label, conditional_input)``
tuples ready for the GFMBench supervised tasks.
"""
import gzip
import logging
import os
import shutil
from ast import literal_eval
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta

from gfmbench_api.utils.fileutils import download_file_from_url
from gfmbench_api.utils.preprocutils import standardize_sequence

LRB_HF_REPO_ID = "InstaDeepAI/genomics-long-range-benchmark"

_ASSEMBLY_URLS = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
}

# Per-split target counts (from the upstream CAGE builder) needed to name the
# final (partial) npz shard correctly.
_CAGE_TOTALS = {"train": 33891, "valid": 2195, "test": 1922}
_CAGE_DEFAULT_LENGTH = 114688  # 896 bins x 128bp
_CAGE_BIN = 128

_EMPTY_COND = np.array([], dtype=np.float32)

Example = Tuple[str, object, np.ndarray]


def ensure_assembly_genome(genome_dir: str, assembly: str) -> str:
    """Download+extract a UCSC reference genome (``hg38``/``hg19``) if missing."""
    os.makedirs(genome_dir, exist_ok=True)
    fa_path = os.path.join(genome_dir, f"{assembly}.fa")
    if os.path.exists(fa_path):
        return fa_path

    url = _ASSEMBLY_URLS[assembly]
    gz_path = fa_path + ".gz"
    logging.info("Downloading reference genome %s (~3GB) from %s ...", assembly, url)
    download_file_from_url(url, gz_path)
    logging.info("Extracting %s ...", os.path.basename(gz_path))
    with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    return fa_path


def _download_lrb_file(filename: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=LRB_HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=cache_dir,
    )


def _get_chromosome(genome: Fasta, chrom):
    """Fetch a chromosome record, tolerating ``chr`` prefix differences."""
    name = str(chrom)
    if name in genome:
        return genome[name]
    alt = name[3:] if name.startswith("chr") else f"chr{name}"
    if alt in genome:
        return genome[alt]
    return None


def builder_pad_sequence(chromosome, start, sequence_length, end=None, negative_strand=False):
    """Center+pad a region to ``sequence_length`` bp (upstream builder logic).

    Returns ``None`` when the padded window falls outside the chromosome, which
    is how the upstream builder drops boundary cases.
    """
    if end:
        pad = (sequence_length - (end - start)) // 2
        start = start - pad
        end = end + pad + (sequence_length % 2)
    else:
        pad = sequence_length // 2
        end = start + pad + (sequence_length % 2)
        start = start - pad

    if start < 0 or end >= len(chromosome):
        return None
    if negative_strand:
        return chromosome[start:end].reverse.complement.seq
    return chromosome[start:end].seq


# ---------------------------------------------------------------------------
# Regulatory elements (enhancer / promoter): binary, single sequence, hg38.
# ---------------------------------------------------------------------------
def build_regulatory_examples(
    cache_dir: str,
    genome_dir: str,
    sequence_length: int,
    element: str,
    subset: bool = True,
    max_num_samples: Optional[int] = None,
) -> Tuple[List[Example], List[Example]]:
    """Return ``(train_examples, test_examples)`` for a regulatory-element task."""
    suffix = "_subset.csv" if subset else ".csv"
    csv_path = _download_lrb_file(f"regulatory_elements/{element}_dataset{suffix}", cache_dir)
    df = pd.read_csv(csv_path)
    genome = Fasta(ensure_assembly_genome(genome_dir, "hg38"), one_based_attributes=False)

    train = _regulatory_split(df, "train", genome, sequence_length, max_num_samples)
    test = _regulatory_split(df, "test", genome, sequence_length, max_num_samples)
    return train, test


def _regulatory_split(df, split, genome, seq_len, n) -> List[Example]:
    sub = df[df["split"] == split]
    if n is not None:
        # Shuffle before truncating: the CSV is grouped by label, so a plain
        # head() would yield a single-class subset (degenerate AUROC/AUPRC).
        sub = sub.sample(frac=1.0, random_state=0).head(n)
    examples: List[Example] = []
    for _, row in sub.iterrows():
        chrom = _get_chromosome(genome, row["CHROM"])
        if chrom is None:
            continue
        seq = builder_pad_sequence(chrom, int(row["START"]) - 1, seq_len, end=int(row["STOP"]) - 1)
        if not seq:
            continue
        examples.append((standardize_sequence(seq), int(row["label"]), _EMPTY_COND))
    return examples


# ---------------------------------------------------------------------------
# Chromatin features (histone marks / DNA accessibility): multi-label, hg19.
# ---------------------------------------------------------------------------
def build_chromatin_examples(
    cache_dir: str,
    genome_dir: str,
    sequence_length: int,
    feature: str,
    subset: bool = True,
    max_num_samples: Optional[int] = None,
) -> Tuple[List[Example], List[Example]]:
    """Return ``(train_examples, test_examples)`` for a chromatin-features task."""
    suffix = "_subset.csv" if subset else ".csv"
    csv_path = _download_lrb_file(f"chromatin_features/histones_and_dnase{suffix}", cache_dir)
    df = pd.read_csv(csv_path)
    label_col = "HISTONES" if "histone" in feature else "DNASE"
    genome = Fasta(ensure_assembly_genome(genome_dir, "hg19"), one_based_attributes=False)

    train = _chromatin_split(df, "train", genome, sequence_length, label_col, max_num_samples)
    test = _chromatin_split(df, "test", genome, sequence_length, label_col, max_num_samples)
    return train, test


def _chromatin_split(df, split, genome, seq_len, label_col, n) -> List[Example]:
    sub = df[df["split"] == split]
    if n is not None:
        # Shuffle before truncating so the capped subset spans both classes
        # across the per-track labels (the CSV is grouped).
        sub = sub.sample(frac=1.0, random_state=0).head(n)
    examples: List[Example] = []
    for _, row in sub.iterrows():
        chrom = _get_chromosome(genome, row["CHROM"])
        if chrom is None:
            continue
        # Centered on the annotated 200bp bin (no explicit end -> symmetric pad).
        seq = builder_pad_sequence(chrom, int(row["POS"]) - 1, seq_len)
        if not seq:
            continue
        labels = np.asarray(literal_eval(row[label_col]), dtype=np.float32)
        examples.append((standardize_sequence(seq), labels, _EMPTY_COND))
    return examples


# ---------------------------------------------------------------------------
# CAGE prediction: binned multi-track regression, hg38.
# ---------------------------------------------------------------------------
def build_cage_examples(
    cache_dir: str,
    genome_dir: str,
    sequence_length: int,
    max_num_samples: Optional[int] = None,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Return ``(train, validation, test)`` examples for CAGE prediction.

    Targets are ``log1p``-transformed (data ships as raw counts) and subset to
    ``sequence_length / 128`` bins, matching the upstream builder.
    """
    coords = pd.read_csv(_download_lrb_file("cage_prediction/sequences_coordinates.csv", cache_dir))
    genome = Fasta(ensure_assembly_genome(genome_dir, "hg38"), one_based_attributes=False)

    train = _cage_split(coords, "train", "train", genome, sequence_length, cache_dir, max_num_samples)
    validation = _cage_split(coords, "validation", "valid", genome, sequence_length, cache_dir, max_num_samples)
    test = _cage_split(coords, "test", "test", genome, sequence_length, cache_dir, max_num_samples)
    return train, validation, test


def _cage_split(coords, split, npz_split, genome, seq_len, cache_dir, n) -> List[Example]:
    sub = coords[coords["split"] == split]
    if sub.empty and split == "validation":
        sub = coords[coords["split"] == "valid"]
    if n is not None:
        sub = sub.head(n)
    if sub.empty:
        return []

    total = _CAGE_TOTALS[npz_split]
    # Download only the npz shards referenced by the selected rows.
    npz_cache = {}
    for floored in sorted({(int(idx) // 1000) * 1000 for idx in sub["npy_idx"]}):
        end = floored + 999 if floored + 1000 <= total else total - 1
        shard = f"cage_prediction/targets_subset/targets-{npz_split}-{floored}-{end}.npz"
        npz_cache[floored] = np.load(_download_lrb_file(shard, cache_dir))

    examples: List[Example] = []
    for _, row in sub.iterrows():
        chrom = _get_chromosome(genome, row["chrom"])
        if chrom is None:
            continue
        seq = builder_pad_sequence(chrom, int(row["start"]) - 1, seq_len, end=int(row["stop"]) - 1)
        if not seq:
            continue
        npy_idx = int(row["npy_idx"])
        floored = (npy_idx // 1000) * 1000
        targets = npz_cache[floored][f"target-{npz_split}-{npy_idx}.npy"][0]  # [896, 50]
        if seq_len < _CAGE_DEFAULT_LENGTH:
            idx_diff = (_CAGE_DEFAULT_LENGTH - seq_len) // 2 // _CAGE_BIN
            targets = targets[idx_diff:-idx_diff]
        targets = np.log1p(np.asarray(targets, dtype=np.float32))
        examples.append((standardize_sequence(seq), targets, _EMPTY_COND))
    return examples
