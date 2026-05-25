#!/usr/bin/env python3
"""One-off helper: rewrite SPDX + per-file third-party URL headers. Not run at runtime."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPDX_CORE = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

URL_PREAMBLE = (
    "# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):\n"
)

NO_URL_LINE = "# This module does not embed third-party data download URLs.\n"

# Canonical URL -> license label (must match THIRD_PARTY_NOTICES.md intent)
U = {
    "gue": ("https://huggingface.co/datasets/leannmlindsey/GUE", "MIT"),
    "lrb_hf": (
        "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark",
        "CC-BY-NC-4.0",
    ),
    "loleve_hf": (
        "https://huggingface.co/datasets/Marks-lab/LOL-EVE-eQTL_benchmark",
        "MIT",
    ),
    "m42": (
        "https://huggingface.co/datasets/m42-health/variant-benchmark",
        "CC-BY-NC-4.0",
    ),
    "songlab_clinvar": ("https://huggingface.co/datasets/songlab/clinvar", "MIT"),
    "traitgym": ("https://huggingface.co/datasets/songlab/TraitGym", "MIT"),
    "hg38": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "LicenseRef-UCSC-Genome-Browser",
    ),
    "bend_expr_bed": (
        "https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_expression.bed",
        "BSD-3-Clause",
    ),
    "bend_dis_bed": (
        "https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_disease.bed",
        "BSD-3-Clause",
    ),
    "biorxiv_vep": (
        "https://www.biorxiv.org/content/biorxiv/early/2025/09/10/2025.09.05.674459/DC1/embed/media-1.zip?download=true",
        "CC-BY-NC-ND-4.0",
    ),
    "ncbi_clinvar": (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz",
        "LicenseRef-NCBI-Data",
    ),
    "brca_xlsx": (
        "https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx",
        "Apache-2.0",
    ),
    "brca_fna": (
        "https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/GRCh37.p13_chr17.fna.gz",
        "Apache-2.0",
    ),
    "dnabert2": ("https://huggingface.co/zhihan1996/DNABERT-2-117M", "Apache-2.0"),
    "dnabert6": ("https://huggingface.co/armheb/DNA_bert_6", "Apache-2.0"),
}


def _h(*keys: str) -> list[tuple[str, str]]:
    return [U[k] for k in keys]


# Relative path from repo root -> list of keys into U
FILE_KEYS: dict[str, tuple[str, ...]] = {
    "gfmbench_api/tasks/concrete/gue_promoter_all_task.py": ("gue",),
    "gfmbench_api/tasks/concrete/gue_splice_site_task.py": ("gue",),
    "gfmbench_api/tasks/concrete/gue_tf_all_task.py": ("gue",),
    "gfmbench_api/tasks/concrete/bend_vep_expression_task.py": ("bend_expr_bed", "hg38"),
    "gfmbench_api/tasks/concrete/bend_vep_disease_task.py": ("bend_dis_bed", "hg38"),
    "gfmbench_api/tasks/concrete/variant_benchmarks_coding_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/variant_benchmarks_common_vs_rare_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/variant_benchmarks_expression_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/variant_benchmarks_meqtl_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/variant_benchmarks_non_coding_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/variant_benchmarks_sqtl_task.py": ("m42",),
    "gfmbench_api/tasks/concrete/lrb_causal_eqtl_task.py": ("lrb_hf", "hg38"),
    "gfmbench_api/tasks/concrete/lrb_pathogenic_omim_task.py": ("lrb_hf", "hg38"),
    "gfmbench_api/tasks/concrete/loleve_causal_eqtl_task.py": ("loleve_hf",),
    "gfmbench_api/tasks/concrete/traitgym_complex_task.py": ("traitgym",),
    "gfmbench_api/tasks/concrete/traitgym_mendelian_task.py": ("traitgym",),
    "gfmbench_api/tasks/concrete/songlab_clinvar_task.py": ("songlab_clinvar", "hg38"),
    "gfmbench_api/tasks/concrete/clinvar_vepeval_task.py": ("biorxiv_vep", "hg38"),
    "gfmbench_api/tasks/concrete/clinvar_indel_task.py": ("ncbi_clinvar", "hg38"),
    "gfmbench_api/tasks/concrete/brca1_task.py": ("brca_xlsx", "brca_fna"),
    "gfmbench_api/utils/fileutils.py": ("hg38",),
    "usage_examples/sanity_models/dna_bert2_model.py": ("dnabert2",),
    "usage_examples/sanity_models/dna_bert_model.py": ("dnabert6",),
}

CODING = "# -*- coding: utf-8 -*-\n"

OLD_HEADER_START = "# SPDX-FileCopyrightText:"


def _is_spdx_header_line(line: str) -> bool:
    s = line.rstrip("\n")
    if s.startswith("# SPDX-FileCopyrightText"):
        return True
    if s.startswith("# SPDX-License-Identifier"):
        return True
    if s == "#":
        return True
    if s.startswith("# Licensed under the Apache License"):
        return True
    if s.startswith("# you may not use this file except"):
        return True
    if s.startswith("# You may obtain a copy of the License"):
        return True
    if s.startswith("# http://www.apache.org/licenses/LICENSE-2.0"):
        return True
    if s.startswith("# Unless required by applicable law"):
        return True
    if s.startswith("# distributed under the License is distributed"):
        return True
    if s.startswith("# WITHOUT WARRANTIES OR CONDITIONS"):
        return True
    if s.startswith("# See the License for the specific language"):
        return True
    if s.startswith("# limitations under the License."):
        return True
    if "Third-party URL notices" in s:
        return True
    if s.startswith("# - http://") or s.startswith("# - https://"):
        return True
    if s == "# This module does not embed third-party data download URLs.":
        return True
    return False


def strip_old_header(text: str) -> tuple[str, str]:
    """Return (coding_prefix, rest_after_header)."""
    coding_prefix = ""
    pos = 0
    if text.startswith(CODING):
        coding_prefix = CODING
        pos = len(CODING)
    body = text[pos:]
    if not body.startswith(OLD_HEADER_START):
        raise ValueError("Expected SPDX header")
    lines = body.splitlines(keepends=True)
    j = 0
    while j < len(lines) and _is_spdx_header_line(lines[j]):
        j += 1
    return coding_prefix, "".join(lines[j:])


def make_new_header(keys: tuple[str, ...]) -> str:
    if not keys:
        return SPDX_CORE + NO_URL_LINE
    pairs = _h(*keys)
    lines = "\n".join(f"# - {u} — {lic}" for u, lic in pairs)
    return SPDX_CORE + URL_PREAMBLE + lines + "\n"


def main() -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if "scripts" in path.parts and path.name.startswith("_apply"):
            continue
        rel = str(path.relative_to(ROOT))
        keys = FILE_KEYS.get(rel, ())
        text = path.read_text(encoding="utf-8")

        try:
            coding_prefix, rest = strip_old_header(text)
        except ValueError:
            print("SKIP (no SPDX):", rel)
            continue

        new_header = make_new_header(keys)
        new_text = coding_prefix + new_header + rest
        path.write_text(new_text, encoding="utf-8", newline="\n")

    print("Done. Updated", len(list(ROOT.rglob("*.py"))) - 1, "files (approx).")


if __name__ == "__main__":
    main()
