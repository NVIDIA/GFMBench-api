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

# This module does not embed third-party data download URLs.
"""Genomic locus columns appended to ``conditional_input`` on request.

Tasks built from genome coordinates can expose, per example, the assembly,
chromosome, 0-based center position and (for variant tasks) the forward-strand
reference and alternate bases. Coordinate-aware models read them back with
``decode_locus_batch``. The columns are only emitted when the task is created
with ``task_config["emit_genomic_locus"] = True``, so models that treat
conditional inputs as features are unaffected by default.

Values are stored as float64: genome positions exceed float32's exact integer
range.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

LOCUS_COLUMNS = (
    "locus_assembly",
    "locus_chrom",
    "locus_center0",
    "locus_ref",
    "locus_alt",
)

ASSEMBLY_CODES = {"GRCh37": 37, "GRCh38": 38}
_ASSEMBLY_NAMES = {code: name for name, code in ASSEMBLY_CODES.items()}

_CHROMOSOMES = [str(index) for index in range(1, 23)] + ["X", "Y", "M"]
CHROM_CODES = {name: index + 1 for index, name in enumerate(_CHROMOSOMES)}
_CHROM_NAMES = {code: f"chr{name}" for name, code in CHROM_CODES.items()}

BASE_CODES = {"A": 0, "C": 1, "G": 2, "T": 3}
_BASE_NAMES = {code: base for base, code in BASE_CODES.items()}
NO_ALLELE = -1

_MAX_POSITION = 2**53


@dataclass(frozen=True)
class GenomicLocus:
    """One example's genome anchor; ``ref``/``alt`` are None without a variant."""

    assembly: str
    chrom: str
    center0: int
    ref: Optional[str] = None
    alt: Optional[str] = None


def chrom_code(chrom: str) -> int:
    name = str(chrom)
    if name.startswith("chr"):
        name = name[3:]
    if name == "MT":
        name = "M"
    if name not in CHROM_CODES:
        raise ValueError(f"Unsupported chromosome {chrom!r} for a genomic locus")
    return CHROM_CODES[name]


def _base_code(base: Optional[str]) -> int:
    # Alleles other than one A/C/G/T base (indels, N) have no code: a pair
    # whose sequences differ but whose alleles are NO_ALLELE is not an SNV.
    if base is None:
        return NO_ALLELE
    return BASE_CODES.get(str(base).upper(), NO_ALLELE)


def encode_locus(locus: GenomicLocus) -> np.ndarray:
    if locus.assembly not in ASSEMBLY_CODES:
        raise ValueError(f"Unsupported assembly {locus.assembly!r}")
    if not 0 <= int(locus.center0) < _MAX_POSITION:
        raise ValueError(f"Locus position {locus.center0} is out of range")
    ref, alt = _base_code(locus.ref), _base_code(locus.alt)
    if NO_ALLELE in (ref, alt):
        ref = alt = NO_ALLELE
    return np.array(
        [
            ASSEMBLY_CODES[locus.assembly],
            chrom_code(locus.chrom),
            int(locus.center0),
            ref,
            alt,
        ],
        dtype=np.float64,
    )


def append_locus(conditional_input: Any, locus: GenomicLocus) -> np.ndarray:
    """Return ``conditional_input`` (possibly empty) followed by the locus columns."""
    base = (
        np.zeros(0, dtype=np.float64)
        if conditional_input is None
        else np.asarray(conditional_input, dtype=np.float64).reshape(-1)
    )
    return np.concatenate((base, encode_locus(locus)))


def locus_metadata_frame(base: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Append the locus rows to a task's conditional-input schema."""
    rows = pd.DataFrame(
        {
            "meta_data_name": list(LOCUS_COLUMNS),
            "data_type": ["integer"] * len(LOCUS_COLUMNS),
            "min_value": [min(ASSEMBLY_CODES.values()), 1, 0, NO_ALLELE, NO_ALLELE],
            "max_value": [
                max(ASSEMBLY_CODES.values()),
                len(CHROM_CODES),
                _MAX_POSITION - 1,
                len(BASE_CODES) - 1,
                len(BASE_CODES) - 1,
            ],
        }
    )
    if base is None or len(base) == 0:
        return rows
    return pd.concat((base, rows), ignore_index=True)


def locus_column_offset(metadata: Optional[pd.DataFrame]) -> Optional[int]:
    """Index of the first locus column, or None when the schema has none."""
    if metadata is None or "meta_data_name" not in metadata:
        return None
    names = [str(name) for name in metadata["meta_data_name"]]
    if tuple(names[-len(LOCUS_COLUMNS):]) != LOCUS_COLUMNS:
        return None
    return len(names) - len(LOCUS_COLUMNS)


def decode_locus(values: Sequence[float]) -> GenomicLocus:
    assembly, chrom, center0, ref, alt = (int(round(float(value))) for value in values)
    return GenomicLocus(
        assembly=_ASSEMBLY_NAMES[assembly],
        chrom=_CHROM_NAMES[chrom],
        center0=center0,
        ref=_BASE_NAMES.get(ref),
        alt=_BASE_NAMES.get(alt),
    )


def decode_locus_batch(
    conditional_input: Any, metadata: Optional[pd.DataFrame]
) -> Optional[List[GenomicLocus]]:
    """Decode a batched ``conditional_input`` ([batch, columns]); None without loci."""
    offset = locus_column_offset(metadata)
    if offset is None or conditional_input is None:
        return None
    if hasattr(conditional_input, "detach"):
        conditional_input = conditional_input.detach().cpu().numpy()
    values = np.asarray(conditional_input, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[-1] != offset + len(LOCUS_COLUMNS):
        raise ValueError(
            f"conditional_input has {values.shape[-1]} columns, expected "
            f"{offset + len(LOCUS_COLUMNS)}"
        )
    return [decode_locus(row[offset:]) for row in values]
