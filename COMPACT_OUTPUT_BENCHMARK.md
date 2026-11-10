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

## Measured precedent

The October 23 pipeline campaign used the same Evo2 1B checkpoint and 8,192
token context on GB300s. It evaluated 23 upstream task implementations with
three warm repetitions per stage: 256 BRCA1 examples and 512 examples for each
other task. Fixed seeded random heads were used for supervised tasks.

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
