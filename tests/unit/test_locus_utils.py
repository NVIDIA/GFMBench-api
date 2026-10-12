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

"""
Unit tests for the genomic locus columns of ``conditional_input``.

Uses in-memory chromosomes and data frames; no reference genome or network.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils import lrb_local
from gfmbench_api.utils.locus import (
    LOCUS_COLUMNS,
    NO_ALLELE,
    GenomicLocus,
    append_locus,
    decode_locus,
    decode_locus_batch,
    encode_locus,
    locus_column_offset,
    locus_metadata_frame,
)
from gfmbench_api.utils.preprocutils import (
    build_forward_centered_loci,
    build_forward_centered_seqs,
    extract_snv_sequences_centered,
)


class _Record:
    """Minimal pyfaidx-like chromosome: ``len``, slicing and ``.seq``."""

    def __init__(self, seq: str) -> None:
        self.seq = seq

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, item):
        return _Record(self.seq[item])

    def __str__(self) -> str:
        return self.seq


class _Genome(dict):
    def keys(self):
        return list(super().keys())


def _chromosome(length: int = 400, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_encode_decode_round_trip_keeps_large_positions_exact():
    locus = GenomicLocus("GRCh38", "chrX", 248_956_421, "a", "T")
    values = encode_locus(locus)
    assert values.dtype == np.float64
    assert values.tolist() == [38, 23, 248_956_421, 0, 3]
    assert decode_locus(values) == GenomicLocus("GRCh38", "chrX", 248_956_421, "A", "T")


def test_reference_window_and_non_snv_alleles_have_no_allele_code():
    window = encode_locus(GenomicLocus("GRCh37", "8", 1_000))
    indel = encode_locus(GenomicLocus("GRCh38", "chrMT", 10, "AC", "A"))
    assert window[3:].tolist() == [NO_ALLELE, NO_ALLELE]
    assert indel.tolist() == [38, 25, 10, NO_ALLELE, NO_ALLELE]
    assert decode_locus(window) == GenomicLocus("GRCh37", "chr8", 1_000)


@pytest.mark.parametrize(
    "locus",
    [
        GenomicLocus("hg38", "chr1", 0),
        GenomicLocus("GRCh38", "chrUn_gl000220", 0),
        GenomicLocus("GRCh38", "chr1", -1),
    ],
)
def test_encode_rejects_unknown_assembly_chromosome_or_position(locus):
    with pytest.raises(ValueError):
        encode_locus(locus)


def test_append_keeps_existing_metadata_first():
    tissue = np.array([7], dtype=np.float32)
    values = append_locus(tissue, GenomicLocus("GRCh38", "chr9", 123, "G", "C"))
    assert values.dtype == np.float64
    assert values.tolist() == [7, 38, 9, 123, 2, 1]
    assert append_locus(np.array([]), GenomicLocus("GRCh38", "chr1", 5)).shape == (5,)


def test_metadata_frame_appends_locus_rows_after_task_rows():
    tissue = pd.DataFrame(
        {"meta_data_name": ["tissue_id"], "data_type": ["integer"], "min_value": [0], "max_value": [48]}
    )
    frame = locus_metadata_frame(tissue)
    assert frame["meta_data_name"].tolist() == ["tissue_id", *LOCUS_COLUMNS]
    assert locus_column_offset(frame) == 1
    assert locus_column_offset(locus_metadata_frame(None)) == 0
    assert locus_column_offset(tissue) is None
    assert locus_column_offset(None) is None


def test_decode_batch_from_collated_tensor():
    rows = [
        append_locus([3], GenomicLocus("GRCh38", "chr2", 10, "A", "G")),
        append_locus([4], GenomicLocus("GRCh38", "chr2", 11, "C", "T")),
    ]
    batch = torch.utils.data.default_collate(rows)
    frame = locus_metadata_frame(
        pd.DataFrame({"meta_data_name": ["tissue_id"], "data_type": ["integer"], "min_value": [0], "max_value": [9]})
    )
    loci = decode_locus_batch(batch, frame)
    assert [locus.center0 for locus in loci] == [10, 11]
    assert [(locus.ref, locus.alt) for locus in loci] == [("A", "G"), ("C", "T")]
    assert decode_locus_batch(batch, None) is None
    with pytest.raises(ValueError):
        decode_locus_batch(batch[:, 1:], frame)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def test_extract_snv_sequences_returns_aligned_loci_only_on_request():
    sequence = _chromosome()
    genome = _Genome(chr1=_Record(sequence))
    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "pos": [101, 201, 301],
            "ref": [sequence[100], "N" if sequence[200] != "N" else "A", sequence[300]],
            "alt": ["A" if sequence[100] != "A" else "C", "G", "T" if sequence[300] != "T" else "G"],
            "label": [0, 1, 1],
        }
    )
    plain = extract_snv_sequences_centered(df, genome, 64, use_logging=False)
    assert len(plain) == 4
    var_seqs, ref_seqs, labels, skipped, loci = extract_snv_sequences_centered(
        df, genome, 64, use_logging=False, locus_assembly="GRCh38"
    )
    assert skipped == 1
    assert len(loci) == len(var_seqs) == len(labels) == 2
    flank = (64 - 1) // 2
    for locus, ref_seq, var_seq in zip(loci, ref_seqs, var_seqs):
        assert locus.assembly == "GRCh38" and locus.chrom == "chr1"
        assert ref_seq[flank] == sequence[locus.center0] == locus.ref
        assert var_seq[flank] == locus.alt
    assert [locus.center0 for locus in loci] == [100, 300]


def test_forward_centered_loci_read_alleles_from_left_sequences():
    df = pd.DataFrame(
        {
            "chrom": ["chr3", "chr4"],
            "position": [1_000, 2_000],
            "ref_left": ["[START_EXON] A C G", "T T A"],
            "alt_left": ["[START_EXON] A C Ṱ", "T T Ĉ"],
            "ref_right": ["T T C", "G G T"],
            "alt_right": ["T T A", "G G G"],
            "label": [0, 1],
        }
    )
    loci = build_forward_centered_loci(df, position_col="position", one_based=True)
    assert loci == [
        GenomicLocus("GRCh38", "chr3", 999, "G", "T"),
        GenomicLocus("GRCh38", "chr4", 1_999, "A", "C"),
    ]
    seqs = build_forward_centered_seqs(df)
    for locus, left, ref_seq, var_seq in zip(loci, ["ACG", "TTA"], seqs["reference_sequence"], seqs["variant_sequence"]):
        assert ref_seq[len(left) - 1] == locus.ref
        assert var_seq[len(left) - 1] == locus.alt
    zero_based = build_forward_centered_loci(df, position_col="position", one_based=False)
    assert [locus.center0 for locus in zero_based] == [1_000, 2_000]


@pytest.mark.parametrize("end_col", ["STOP", None])
def test_lrb_examples_anchor_the_window_center(end_col):
    sequence = _chromosome(2_000, seed=1)
    genome = _Genome(chr8=_Record(sequence))
    df = pd.DataFrame(
        {"CHROM": ["chr8", "chr8"], "START": [801, 1_201], "STOP": [1_001, 1_401], "split": ["test", "test"], "label": [1, 0]}
    )
    seq_len = 513
    examples = lrb_local._rows_to_examples(
        df, "test", genome, seq_len, None,
        start_col="START", end_col=end_col,
        label_fn=lambda row: int(row["label"]),
        assembly="GRCh38", conditional=append_locus,
    )
    assert len(examples) == 2
    for (seq, _, cond), (_, row) in zip(examples, df.iterrows()):
        locus = decode_locus(cond)
        start = int(row["START"]) - 1
        end = int(row["STOP"]) - 1 if end_col else None
        window_start, window_end = lrb_local._padded_window(start, seq_len, end)
        assert window_end - window_start == seq_len
        assert locus.center0 == window_start + seq_len // 2
        assert seq[seq_len // 2] == sequence[locus.center0]
        assert (locus.ref, locus.alt) == (None, None)
    plain = lrb_local._rows_to_examples(
        df, "test", genome, seq_len, None,
        start_col="START", end_col=end_col,
        label_fn=lambda row: int(row["label"]),
        assembly="GRCh38",
    )
    assert all(example[2].size == 0 for example in plain)


# ---------------------------------------------------------------------------
# Zero-shot SNV metrics forward conditional_input
# ---------------------------------------------------------------------------

class _RecordingModel:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def sequence_pos_to_prob_pos(self, sequences, pos, *conditional):
        self.calls.append(("pos", len(conditional)))
        return np.full(len(sequences), pos)

    def infer_masked_sequence_to_token_probs(self, sequences, pos, variant, reference, *conditional):
        self.calls.append(("masked", len(conditional)))
        return np.full(len(sequences), 0.5), np.full(len(sequences), 0.5)


class _StubSNVTask(BaseGFMZeroShotSNVTask):
    def __init__(self, emit: bool) -> None:
        self.emit_genomic_locus = emit
        self.disable_safe_model_call = True

    def get_task_name(self) -> str:
        return "stub_snv"

    def use_reference_cache(self) -> bool:
        return False

    def _create_test_dataset(self):
        return []

    def _get_default_max_seq_len(self) -> int:
        return 3

    def _get_variant_position_in_sequence(self) -> int:
        return 1

    def get_conditional_input_meta_data_frame(self):
        return None


def _snv_task(emit: bool) -> BaseGFMZeroShotSNVTask:
    return _StubSNVTask(emit)


@pytest.mark.parametrize("emit", [False, True])
def test_snv_metrics_forward_conditional_input_only_for_locus_tasks(emit):
    model = _RecordingModel()
    outputs = {"conditional_input": torch.zeros(2, 5, dtype=torch.float64)}
    _snv_task(emit)._update_additional_metrics(
        [], model, ["AAA", "ACA"], ["AGA", "AGA"], np.array([0, 1]), outputs
    )
    expected = 1 if emit else 0
    assert model.calls == [("pos", expected), ("pos", expected), ("masked", expected)]
