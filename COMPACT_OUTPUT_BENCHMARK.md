# Evo2 compact outputs: benchmark note

**Review status:** the implementation and CPU checks are ready. A fresh GB300
timing run of this branch has not yet completed, so the numbers below are
historical evidence, not a measured speedup of this commit.

## Change

Evo2's full sequence API transfers a `[batch, tokens, hidden]` embedding array
to host memory even when a task uses only pooled embeddings or one SNV position.
`--compact_outputs` lets the model reduce or select those values on the GPU.
It is off by default; other models and callers of the full API keep their
existing behavior. Binned regression is outside this compact path.

## Historical measurement setup

The October 23 campaign used the original Evo2 1B checkpoint parameters and an
8,192-token maximum context on NVIDIA GB300s. Backbone autocast was BF16, with
the original FP32 residual and scoring behavior. The qualification receipt
records PyTorch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, and checkpoint
archive SHA-256 `ea4a3f5c9c26d5edc10bdc85165c090ad0ff23ac2670d4f61244f5f0d9d5e817`.
Both native Evo2 and CUDA Fyena used a pinned modern BioNeMo/MCore loading port,
with task inputs staged on local NVMe. This was not the legacy NeMo2 paper
environment. The campaign
evaluated 23 upstream tasks: 256 BRCA1 examples and 512 examples for each other
task. Supervised tasks used fixed seeded random heads; no heads were trained.

Each task and model path started with every optimization off, batch size 32 and
zero loader workers. Accepted stages then applied numerical-gated batch tuning,
compact outputs, other pipeline switches, final batch tuning and a 0/2/4/8
worker comparison. Every final selection used zero workers. Times are medians
of three warm, unprofiled task evaluations per stage. They exclude model startup,
warmup and writing saved outputs. Source construction was measured separately.

## Historical results

| Historical result | Native Evo2 | CUDA Fyena |
|:--|--:|--:|
| Median original-to-final task speedup | **2.02×** | **2.26×** |
| Within-implementation sweeps passing numerical gates | 23/23 | 23/23 |

Across all 46 sweeps, reducing outputs on the GPU before host transfer gave a
**2.065× median incremental speedup**. Batch tuning contributed a 1.010×
median initial gain; worker prefetch was slower. The final speedups combine
several cumulative switches, so they must not be attributed to this PR alone.
The first CAGE implementation failed its output gate; the campaign accepted a
repaired FP32 reduction order. This PR does not change CAGE's binned path.

### All 23 tasks

Times are **baseline → final seconds** within each model path; `×` is baseline
divided by final time. `N/C batch` gives the selected final native/CUDA batch
sizes. Both baselines used batch 32. These rows include other cumulative
optimizations and are **not a timing of the code in this PR**.

| Task | Examples | Native, s | Native × | CUDA Fyena, s | CUDA × | N/C batch |
|:--|--:|--:|--:|--:|--:|--:|
| bend_variant_effects_disease | 512 | 5.078 → 2.688 | 1.89 | 4.457 → 2.075 | 2.15 | 32/512 |
| bend_variant_effects_expression | 512 | 5.083 → 2.681 | 1.90 | 4.476 → 2.062 | 2.17 | 32/512 |
| brca1 | 256 | 41.469 → 20.301 | 2.04 | 38.695 → 16.474 | 2.35 | 32/32 |
| clinvar_indel | 512 | 83.344 → 41.871 | 1.99 | 77.748 → 33.727 | 2.31 | 32/32 |
| gue_promoter_all | 512 | 1.523 → 0.862 | 1.77 | 1.284 → 0.614 | 2.09 | 64/256 |
| gue_splice_site | 512 | 2.158 → 1.212 | 1.78 | 1.778 → 0.815 | 2.18 | 64/512 |
| gue_transcription_factor | 512 | 0.661 → 0.333 | 1.98 | 0.490 → 0.214 | 2.29 | 256/512 |
| loleve_causal_eqtl | 512 | 13.487 → 5.274 | 2.56 | 11.926 → 4.009 | 2.97 | 64/512 |
| lrb_cage_prediction | 512 | 45.687 → 20.868 | 2.19 | 38.196 → 16.889 | 2.26 | 32/32 |
| lrb_chromatin_features_histone_marks | 512 | 41.191 → 20.909 | 1.97 | 36.389 → 16.920 | 2.15 | 32/32 |
| lrb_regulatory_element_enhancer | 512 | 43.794 → 20.816 | 2.10 | 36.386 → 16.912 | 2.15 | 32/32 |
| lrb_variant_effect_causal_eqtl | 512 | 83.363 → 41.669 | 2.00 | 72.770 → 33.720 | 2.16 | 32/32 |
| lrb_variant_effect_pathogenic_omim | 512 | 82.530 → 41.771 | 1.98 | 76.953 → 33.859 | 2.27 | 32/32 |
| songlab_clinvar | 512 | 87.971 → 41.490 | 2.12 | 72.678 → 33.779 | 2.15 | 32/32 |
| traitgym_complex | 512 | 83.419 → 41.790 | 2.00 | 76.953 → 34.112 | 2.26 | 32/64 |
| traitgym_mendelian | 512 | 84.194 → 41.740 | 2.02 | 72.487 → 33.851 | 2.14 | 32/32 |
| var_bench_coding_pathogenicity | 512 | 14.323 → 6.115 | 2.34 | 12.434 → 4.155 | 2.99 | 64/256 |
| var_bench_common_vs_rare | 512 | 14.430 → 6.022 | 2.40 | 12.310 → 4.158 | 2.96 | 64/256 |
| var_bench_expression | 512 | 14.203 → 6.063 | 2.34 | 12.573 → 4.165 | 3.02 | 64/256 |
| var_bench_meqtl | 512 | 14.360 → 6.022 | 2.38 | 12.644 → 4.149 | 3.05 | 64/256 |
| var_bench_non_coding_pathogenicity | 512 | 14.230 → 6.078 | 2.34 | 12.512 → 4.153 | 3.01 | 64/256 |
| var_bench_sqtl | 512 | 14.293 → 6.001 | 2.38 | 12.480 → 4.135 | 3.02 | 64/256 |
| vepeval_clinvar | 512 | 82.846 → 41.683 | 1.99 | 73.180 → 33.761 | 2.17 | 32/32 |

The campaign compared each implementation with its **own** baseline. Native
Evo2 and CUDA Fyena failed strict cross-model equivalence on all 23 tasks;
these timing results make no prediction-quality or paper-score claim.

## Fresh rerun required for this branch

Use the original 23-task fixture order, checkpoint, context, fixed heads,
batch-32 baseline, 256/512 example counts, and three unprofiled warm
repetitions. Compare `compact_outputs=False` and `True` with the same model
runtime and inputs. Save raw probabilities, selected or pooled outputs, scores,
metrics, process logs and peak GPU memory. Require the original within-model
array (`rtol=2e-6`, `atol=2e-6`), probability (`atol=2e-8`), and metric
(`atol=1e-4`) gates before reporting speedups. Report the CAGE task separately
because this branch leaves its output path unchanged.

**Local checks:** 34 targeted unit and end-to-end tests passed;
`ruff check --ignore E402` and `git diff --check` passed. E402 is an existing
import-order issue in the example CLI. No GPU timing or full-model numerical
qualification of this branch is claimed by those checks.

Historical source: `gfmbench-optimization-20261022T093201Z/operator-review-v35`
in the Fyena evidence workspace; the original timing rows and validation
receipts remain archived there.
