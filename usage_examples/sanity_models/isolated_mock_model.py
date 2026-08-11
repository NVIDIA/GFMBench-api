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
"""Example: run model inference in a separate env and process.

Purpose
-------
Show how to keep GFMBench-API tasks/metrics in the host environment
(``basic_requirements.txt``) while running model inference in another
Python env — useful when model deps conflict with the host stack.

What happens
------------
* **Host** (this process): ``IsolatedMockModel`` implements the usual
  ``infer_*`` API and only marshals JSON over stdin/stdout.
* **Worker** (subprocess): each ``infer_*`` call spawns
  ``<worker_python> this_file.py --worker``, which imports worker-only
  packages and returns mock arrays.
* On the **first** call, the host prints a host→worker environment
  switch (interpreters, package versions, PID) so isolation is visible.

This example returns **random** tensors (no real model). The worker uses
``randomgen`` (not in ``basic_requirements.txt`` / host env) to generate
all values — proof that inference ran in the worker env.

Setup (once)
------------
1. Create the worker env (any path; default below)::

       conda create -y -p /opt/conda/envs/gfmbench-isolated-mock python=3.10
       /opt/conda/envs/gfmbench-isolated-mock/bin/pip install \\
           numpy==1.26.4 torch==2.2.2 randomgen

   Worker pins used here: Python 3.10, numpy 1.26.4, torch 2.2.2,
   randomgen 2.3.0 (host may differ, e.g. Python 3.11 / numpy 2.x).

2. Point the adapter at your interpreter: set ``DEFAULT_ISOLATED_PYTHON``
   below, or pass ``worker_python="/path/to/env/bin/python"`` to
   ``IsolatedMockModel(...)``.

Run
---
Registered as ``--model IsolatedMock``::

    python usage_examples/run_benchmark.py \\
      --model IsolatedMock \\
      --linear_prob \\
      --sanity_check_mode \\
      --root_data_dir_path /path/to/data \\
      --csv_path /tmp/isolated_mock.csv

Quick check::

    # Should fail in the host env:
    python -c "import randomgen"
    # Should succeed in the worker env:
    /opt/conda/envs/gfmbench-isolated-mock/bin/python -c \\
        "import randomgen; print(randomgen.__version__)"
"""

from __future__ import annotations

import json
import os
import sys

DEFAULT_ISOLATED_PYTHON = "/opt/conda/envs/gfmbench-isolated-mock/bin/python"
DEFAULT_HIDDEN_DIM = 32
DEFAULT_NUM_LABELS = 2


# ---------------------------------------------------------------------------
# Worker helpers (stdlib-only at definition time; worker deps imported inside)
# ---------------------------------------------------------------------------

def _softmax_rows(x):
    import numpy as np

    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _dispatch(method: str, args: dict, rng):
    import numpy as np

    if method == "infer_sequence_to_labels_probs":
        n = len(args["sequences"])
        k = int(args.get("num_labels", DEFAULT_NUM_LABELS))
        logits = rng.standard_normal((n, k))
        return _softmax_rows(logits).tolist()

    if method == "infer_sequence_to_sequence":
        sequences = args["sequences"]
        n = len(sequences)
        hidden = int(args.get("hidden_dim", DEFAULT_HIDDEN_DIM))
        max_len = int(args.get("max_length", 128))
        seq_lens = [min(len(s), max_len) for s in sequences]
        L = max(seq_lens) if seq_lens else 1
        probs = rng.random((n, L))
        embeds = rng.standard_normal((n, L, hidden))
        reprs = rng.standard_normal((n, hidden))
        return [probs.tolist(), embeds.tolist(), reprs.tolist()]

    if method == "sequence_pos_to_prob_pos":
        sequences = args["sequences"]
        pos = int(args["pos"])
        return [pos if 0 <= pos < len(s) else -1 for s in sequences]

    if method == "infer_masked_sequence_to_token_probs":
        n = len(args["sequences"])
        return [rng.random(n).tolist(), rng.random(n).tolist()]

    if method == "infer_variant_ref_sequences_to_labels_probs":
        n = len(args["variant_sequences"])
        k = int(args.get("num_labels", DEFAULT_NUM_LABELS))
        logits = rng.standard_normal((n, k))
        return _softmax_rows(logits).tolist()

    if method == "infer_sequence_to_regression":
        n = len(args["sequences"])
        num_outputs = int(args.get("num_outputs", 1))
        return rng.standard_normal((n, num_outputs)).tolist()

    raise ValueError(f"Unknown worker method: {method}")


def _worker_main() -> None:
    """Read one JSON request from stdin; write one JSON response to stdout.

    Imports ``randomgen`` here — this package exists only in the isolated env.
    """
    import randomgen  # worker-only package (absent from host / basic_requirements)
    import numpy as np
    import torch as worker_torch

    request = json.loads(sys.stdin.read())
    method = request["method"]
    args = request.get("args", {})

    # All mock values come from randomgen.PCG64 via numpy Generator.
    bit_gen = randomgen.PCG64()
    rng = np.random.Generator(bit_gen)

    proof = {
        "randomgen": getattr(randomgen, "__version__", "unknown"),
        "numpy": np.__version__,
        "torch": worker_torch.__version__,
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "executable": sys.executable,
    }

    try:
        result = _dispatch(method, args, rng)
        sys.stdout.write(json.dumps({"ok": True, "result": result, "proof": proof}))
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001 — surface any worker failure to proxy
        sys.stdout.write(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "proof": proof}
            )
        )
        sys.stdout.write("\n")
        raise


# Enter worker mode before importing host-only deps (gfmbench_api / torch.nn).
if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main()
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Host proxy (imported from sense-env-att / caller environment)
# ---------------------------------------------------------------------------

import logging
import subprocess
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel

logger = logging.getLogger(__name__)


class IsolatedMockModel(BaseGFMModel):
    """Host-side proxy: each inference call runs in the isolated worker env."""

    def __init__(
        self,
        device: str = "cpu",
        max_length: int = 128,
        worker_python: str = DEFAULT_ISOLATED_PYTHON,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_labels: int = DEFAULT_NUM_LABELS,
        **_kwargs,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.worker_python = worker_python
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._logged_proof = False

        if not os.path.isfile(self.worker_python):
            raise RuntimeError(
                f"Isolated worker Python not found: {self.worker_python}\n"
                "Create the env as documented in the module docstring of "
                "usage_examples/sanity_models/isolated_mock_model.py"
            )

    def _call(self, method: str, args: dict) -> dict:
        request = {"method": method, "args": args}
        proc = subprocess.run(
            [self.worker_python, os.path.abspath(__file__), "--worker"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"IsolatedMock worker failed (exit {proc.returncode}).\n"
                f"stderr:\n{proc.stderr}\n"
                f"stdout:\n{proc.stdout}\n"
                "Ensure /opt/conda/envs/gfmbench-isolated-mock exists with "
                "numpy==1.26.4, torch==2.2.2, and randomgen "
                "(see module docstring)."
            )
        try:
            response = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(
                f"IsolatedMock worker returned invalid JSON.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            ) from exc
        if not response.get("ok", False):
            raise RuntimeError(
                f"IsolatedMock worker error: {response.get('error', response)}"
            )
        proof = response.get("proof")
        if proof and not self._logged_proof:
            host_exe = sys.executable
            worker_exe = proof.get("executable", self.worker_python)
            msg = (
                "\n"
                "[IsolatedMock] Environment switch for model inference\n"
                f"  Host (tasks / gfmbench_api):  {host_exe}\n"
                f"    python={sys.version.split()[0]}  "
                f"numpy={np.__version__}  torch={torch.__version__}\n"
                f"  Worker (model inference):     {worker_exe}\n"
                f"    python={proof.get('python')}  "
                f"numpy={proof.get('numpy')}  torch={proof.get('torch')}  "
                f"randomgen={proof.get('randomgen')}  pid={proof.get('pid')}\n"
                "  (Subsequent infer_* calls reuse this worker env; "
                "each call is a new subprocess.)\n"
            )
            logger.info(msg)
            print(msg, flush=True)
            self._logged_proof = True
        return response

    def _as_array(self, value) -> Optional[np.ndarray]:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    def infer_sequence_to_labels_probs(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        resp = self._call(
            "infer_sequence_to_labels_probs",
            {
                "sequences": list(sequences),
                "hidden_dim": self.hidden_dim,
                "num_labels": self.num_labels,
            },
        )
        return self._as_array(resp["result"])

    def infer_sequence_to_sequence(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        resp = self._call(
            "infer_sequence_to_sequence",
            {
                "sequences": list(sequences),
                "hidden_dim": self.hidden_dim,
                "max_length": self.max_length,
            },
        )
        probs, embeds, reprs = resp["result"]
        return self._as_array(probs), self._as_array(embeds), self._as_array(reprs)

    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        resp = self._call(
            "sequence_pos_to_prob_pos",
            {"sequences": list(sequences), "pos": int(pos)},
        )
        return np.asarray(resp["result"], dtype=np.int64)

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        resp = self._call(
            "infer_masked_sequence_to_token_probs",
            {
                "sequences": list(sequences),
                "variant_pos": int(variant_pos),
                "variant_letters": list(variant_letters),
                "reference_letters": list(reference_letters),
            },
        )
        var_p, ref_p = resp["result"]
        return self._as_array(var_p), self._as_array(ref_p)

    def infer_variant_ref_sequences_to_labels_probs(
        self,
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        resp = self._call(
            "infer_variant_ref_sequences_to_labels_probs",
            {
                "variant_sequences": list(variant_sequences),
                "ref_sequences": list(ref_sequences),
                "num_labels": self.num_labels,
            },
        )
        return self._as_array(resp["result"])

    def infer_sequence_to_regression(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        resp = self._call(
            "infer_sequence_to_regression",
            {"sequences": list(sequences), "num_outputs": 1},
        )
        return self._as_array(resp["result"])

    def _sequence_to_representative(self, sequences: List[str]) -> torch.Tensor:
        """Return sequence reps as a host torch tensor (for linear probing)."""
        _, _, reprs = self.infer_sequence_to_sequence(sequences)
        if reprs is None:
            raise RuntimeError("IsolatedMock returned None sequence_representative")
        return torch.from_numpy(np.asarray(reprs, dtype=np.float32)).to(self.device)

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        yield self._dummy_param

    def eval(self) -> "IsolatedMockModel":
        return self

    def train(self, mode: bool = True) -> "IsolatedMockModel":
        return self

    def to(self, device: str) -> "IsolatedMockModel":
        self.device = device
        self._dummy_param = self._dummy_param.to(device)
        return self

    def load_checkpoint(self, checkpoint_path: str) -> "IsolatedMockModel":
        # Mock model has no weights to load.
        return self


if __name__ == "__main__":
    print(
        "IsolatedMockModel adapter. Import IsolatedMockModel from the host env, "
        "or run with --worker under the isolated env Python.\n"
        f"Default worker: {DEFAULT_ISOLATED_PYTHON}",
        file=sys.stderr,
    )
    raise SystemExit(2)
