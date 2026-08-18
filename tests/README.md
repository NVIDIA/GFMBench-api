# Tests

Test suite for gfmbench-api.

## Specs (install & pytest)

| Spec | Value |
|------|--------|
| Python | `>=3.10` (same as the package) |
| Test runner | `pytest>=8.0` |
| Coverage (optional) | `pytest-cov` |
| Test root | `tests/` |
| Import path | repo root on `PYTHONPATH` (so `gfmbench_api` and `usage_examples` resolve) |

Install the package (editable), then — **for developers who run the pytest suite only** — the test tools:

```bash
pip install -e .
pip install "pytest>=8.0" pytest-cov
```

From the **repository root**:

```bash
pytest tests/
# or, if imports fail without editable install:
PYTHONPATH=. pytest tests/
```

Optional coverage:

```bash
pytest tests/ --cov=gfmbench_api --cov-report=term-missing
```

## Test files

| File | What it checks | Network | GPU |
|------|----------------|---------|-----|
| `tests/unit/test_caching_utils.py` | `SequenceInferenceCache` semantics — cache hits/misses, dedup, disable, clear, size limits, output type fidelity, key correctness, variable-length padding | No | No |
| `tests/e2e/test_smoke.py` | Full eval pipeline with `MockGFMModel` and local fixture CSVs | No | No |
| `tests/e2e/test_download.py` | Tasks download data into an empty temp directory | Yes | No |
| `tests/e2e/test_heavy.py` | Real DNABERT2 benchmark on all tasks (sanity mode); scores compared to pinned baseline CSV | Yes | Recommended |

### Unit (`tests/unit/test_caching_utils.py`)

Thirteen tests for `SequenceInferenceCache`, grouped into five areas:

| Group | Tests |
|-------|-------|
| Core semantics | Full cache hit skips `fn`; partial hit calls `fn` for misses only; duplicate sequences deduplicated within a batch |
| Disable and clear | `disable=True` bypasses read and write; `clear()` invalidates all entries and resets byte accounting |
| Size limits | `max_size_gb=0` disables caching; positive cap stops writes when full while preserving hits; partial batch stores only entries that fit |
| Output type fidelity | Torch tensor round-trip preserves dtype and device; tuple output `(ndarray, ndarray, None)` has all slots restored |
| Key and merge correctness | Different scalar extra args produce separate cache entries; variable-length 2-D embeddings are zero-padded on merge |

No model, network, or GPU required — all tests use `Mock` and plain NumPy arrays (the Torch test skips automatically if PyTorch is not installed).

### Smoke (`tests/e2e/test_smoke.py`)

Six tests covering supervised single-seq, supervised variant, zero-shot SNV/indel, CSV report pipeline, and graceful handling of missing model methods. Uses tiny local fixtures (`tests/fixtures/data/`), 8 samples per task.

### Download (`tests/e2e/test_download.py`)

Parametrized over all tasks in `TASK_REGISTRY` except `vepeval_clinvar`, plus one idempotent cache-reuse test for GUE. Each task is initialized from a shared empty temp directory; downloaded files under the task data directory (and shared `reference_genome/hg38.fa` when applicable) must be non-empty. `loleve_causal_eqtl` loads via the HuggingFace hub cache only — that task is validated via a non-empty test split instead of on-disk artifacts under `data_root`.

### Heavy (`tests/e2e/test_heavy.py`)

Two tests sharing the same DNABERT2 sanity config (all tasks in `TASK_REGISTRY`, 100 samples each, linear probe, 1 epoch):

| Test | Purpose |
|------|---------|
| `test_heavy_sanity_regression` | Run benchmark and compare to `tests/fixtures/dnabert2_sanity_baseline.csv` |
| `test_heavy_update_baseline` | Run benchmark and overwrite the pinned baseline CSV |

Heavy tests download task data into a temp directory automatically.

## Running tests

Run the full suite:

```bash
pytest tests/
```

Run by file or test name:

```bash
pytest tests/unit/test_caching_utils.py
pytest tests/e2e/test_smoke.py
pytest tests/e2e/test_download.py
pytest tests/e2e/test_heavy.py::test_heavy_sanity_regression
```

### Heavy baseline

Pinned scores live in `tests/fixtures/` (e.g. `dnabert2_sanity_baseline.csv`). Each row: `task`, `metric`, `expected`, `atol` (default tolerance 0.02).

Regenerate after an intentional model or pipeline change:

```bash
pytest tests/e2e/test_heavy.py::test_heavy_update_baseline -s
```

The update test is skipped during `pytest tests/` unless you invoke it explicitly by name.
