# Third-Party Notices and Attributions

This document lists third-party open source software, models, datasets, and data sources used with or pulled in by **GFMBench-API**. SPDX License List short identifiers are used where applicable. Custom or site-specific terms are indicated with `LicenseRef-*` (see project `LICENSE` and upstream sites for full text).
---

## Python packages (direct dependencies in `requirements.txt`)

| Component | SPDX or license identifier | Notes |
|-----------|----------------------------|--------|
| aiohappyeyeballs | PSF-2.0 | |
| aiohttp | Apache-2.0 | |
| aiosignal | Apache-2.0 | |
| anyio | MIT | |
| attrs (python-attrs) | MIT | |
| biopython | LicenseRef-Biopython | Biopython License Agreement |
| certifi (python-certifi) | ISC | |
| charset-normalizer | MIT | |
| click | BSD-3-Clause | |
| datasets | Apache-2.0 | |
| dill (python-dill) | MIT | |
| einops | MIT | |
| filelock | Unlicense | |
| frozenlist | Apache-2.0 | |
| fsspec (filesystem_spec) | BSD-3-Clause | |
| h11 | MIT | |
| hf-xet | Apache-2.0 | |
| httpcore | BSD-3-Clause | |
| httpx | BSD-3-Clause | |
| huggingface_hub | Apache-2.0 | |
| idna | BSD-3-Clause | |
| Jinja2 (jinjapython) | BSD-3-Clause | |
| joblib | BSD-3-Clause | |
| MarkupSafe | BSD-3-Clause | |
| mpmath | BSD-3-Clause | |
| multidict | Apache-2.0 | |
| multiprocess (multi_process) | BSD-3-Clause | |
| networkx | BSD-3-Clause | |
| numpy | BSD-3-Clause | |
| pandas | BSD-3-Clause | |
| propcache | Apache-2.0 | |
| pyarrow | Apache-2.0 | |
| pyfaidx | BSD-3-Clause | |
| python-dateutil | Apache-2.0 | |
| pytz | MIT | |
| PyYAML | MIT | |
| regex | CNRI-Python AND Apache-2.0 | Dual-licensed; check upstream |
| requests (psf-requests) | Apache-2.0 | |
| safetensors | Apache-2.0 | |
| scikit-learn | BSD-3-Clause | |
| scipy | BSD-3-Clause | |
| shellingham | ISC | |
| six | MIT | |
| sympy | BSD-3-Clause | |
| threadpoolctl | BSD-3-Clause | |
| tokenizers | Apache-2.0 | |
| torch | BSD-3-Clause | PyTorch; see [PyTorch license](https://github.com/pytorch/pytorch/blob/master/LICENSE) |
| tqdm | MIT AND MPL-2.0 | Dual-licensed; check upstream |
| transformers | Apache-2.0 | |
| triton | MIT | |
| typer-slim | MIT | |
| typing_extensions | PSF-2.0 | |
| tzdata | Apache-2.0 | |
| urllib3 | MIT | |
| xxhash | BSD-3-Clause | |
| yarl | Apache-2.0 | |

---

## Hugging Face models and datasets (URLs)

| Resource | URL | SPDX or license identifier |
|----------|-----|----------------------------|
| DNABERT-2-117M | https://huggingface.co/zhihan1996/DNABERT-2-117M | Apache-2.0 |
| DNA_bert_6 | https://huggingface.co/armheb/DNA_bert_6 | Apache-2.0 |
| GUE | https://huggingface.co/datasets/leannmlindsey/GUE | MIT |
| Genomics Long Range Benchmark | https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark | CC-BY-NC-4.0 |
| LOL-EVE eQTL benchmark | https://huggingface.co/datasets/Marks-lab/LOL-EVE-eQTL_benchmark | MIT |
| variant-benchmark | https://huggingface.co/datasets/m42-health/variant-benchmark | CC-BY-NC-4.0 |
| songlab/clinvar | https://huggingface.co/datasets/songlab/clinvar | MIT |
| TraitGym | https://huggingface.co/datasets/songlab/TraitGym | MIT |

---

## Reference genomes and public data downloads (URLs)

| Resource | URL | SPDX or license identifier |
|----------|-----|----------------------------|
| Human hg38 (full) | https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | LicenseRef-UCSC-Genome-Browser |
| Human hg38 (chromosomes) | https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/ | LicenseRef-UCSC-Genome-Browser |
| Mouse mm39 | https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz | LicenseRef-UCSC-Genome-Browser |
| Zebrafish danRer11 | https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz | LicenseRef-UCSC-Genome-Browser |
| Drosophila dm6 | https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz | LicenseRef-UCSC-Genome-Browser |
| C. elegans ce11 | https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz | LicenseRef-UCSC-Genome-Browser |
| Arabidopsis TAIR10 | https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz | LicenseRef-Ensembl-Data |
| ClinVar variant summary | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz | LicenseRef-NCBI-Data |
| Supplementary media (BEND VEP) | https://www.biorxiv.org/content/biorxiv/early/2025/09/10/2025.09.05.674459/DC1/embed/media-1.zip?download=true | CC-BY-NC-ND-4.0 |
| Variant effects expression BED | https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_expression.bed | BSD-3-Clause |
| Variant effects disease BED | https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_disease.bed | BSD-3-Clause |
| BRCA1 supplementary XLSX | https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx | Apache-2.0 |
| BRCA1 GRCh37 chr17 FASTA | https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/GRCh37.p13_chr17.fna.gz | Apache-2.0 |

---

## LicenseRef notes

- **LicenseRef-UCSC-Genome-Browser:** https://genome.ucsc.edu/license/
- **LicenseRef-Ensembl-Data:** Use is subject to Ensembl / EMBL-EBI terms; see https://www.ensembl.org/info/about/legal/index.html
- **LicenseRef-NCBI-Data:** NCBI policies apply; see https://www.ncbi.nlm.nih.gov/home/about/policies/
- **LicenseRef-Biopython:** https://github.com/biopython/biopython/blob/master/LICENSE.rst

Review all upstream license and attribution requirements before redistribution or commercial use.
