"""GFM-Bench adapter for BioNeMo Evo2.

Loads an Evo2 checkpoint using ``bionemo.core.data.load`` (handles NGC
download + caching) and NeMo's ``io.load_context`` (rebuilds the exact
model config stored alongside the checkpoint).  Then exposes the
duck-typed interface that GFM-Bench tasks expect.

Notebook-compatible: initialises Megatron parallel state manually
instead of going through the NeMo Trainer.

Available checkpoint tags (pass as ``ckpt_tag``):
    evo2/1b-8k-bf16:1.0   — 1B params, 8K context, bf16
    evo2/1b-8k:1.0         — 1B params, 8K context, fp8
    evo2/7b-8k:1.0         — 7B params, 8K context
    evo2/7b-1m:1.0         — 7B params, 1M context
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.dist_checkpointing import load as dist_ckpt_load
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Megatron init (single-GPU, notebook-safe)
# ---------------------------------------------------------------------------

def _ensure_megatron_initialized() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(42)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class Evo2BioNeMoModel:
    """BioNeMo Evo2 → GFM-Bench adapter.

    Parameters
    ----------
    ckpt_tag : str
        A BioNeMo resource tag such as ``"evo2/1b-8k-bf16:1.0"`` or
        ``"evo2/7b-8k:1.0"``.  ``bionemo.core.data.load`` downloads and
        caches the NeMo2 checkpoint from NGC automatically.
        Alternatively, a local directory path to an already-extracted
        NeMo2 checkpoint.
    max_length : int
        Maximum token length for inference (sequences are truncated).
    """

    force_linear_probe = True
    supports_backbone_finetuning = False

    def __init__(
        self,
        ckpt_tag: str = "evo2/1b-8k-bf16:1.0",
        max_length: int = 8192,
        device: str = "cuda",
    ) -> None:
        self.max_length = max_length
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Evo2BioNeMoModel requires a CUDA device.")

        # ── 1. Megatron init ──────────────────────────────────────────────
        _ensure_megatron_initialized()

        # ── 2. Resolve checkpoint path ────────────────────────────────────
        ckpt_root = Path(ckpt_tag)
        if not ckpt_root.is_dir():
            from bionemo.core.data.load import load
            logger.info("Downloading/caching checkpoint: %s", ckpt_tag)
            ckpt_root = load(ckpt_tag)
        logger.info("Checkpoint root: %s", ckpt_root)

        # ── 3. Rebuild model from checkpoint config ───────────────────────
        from nemo.lightning import io as nemo_io
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        nemo_model = nemo_io.load_context(path=ckpt_root / "context", subpath="model")
        nemo_model.configure_model()  # creates nemo_model.module (the Megatron-Core model)
        self.model = nemo_model.module
        self.model = self.model.to(self.device).bfloat16().eval()

        self.hidden_size = nemo_model.config.hidden_size
        self.hidden_dim = self.hidden_size

        self.tokenizer = get_nmt_tokenizer("byte-level")
        self.vocab_size = self.tokenizer.vocab_size

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model created: %s parameters, hidden=%d", f"{n_params:,}", self.hidden_size)

        # ── 4. Load checkpoint weights ────────────────────────────────────
        #   NeMo2 checkpoints store keys with "module." prefix
        sd = self.model.sharded_state_dict(prefix="module.")
        loaded = dist_ckpt_load(
            sd, str(ckpt_root / "weights"),
            sharded_strategy=TorchDistLoadShardedStrategy(),
        )
        if isinstance(loaded, tuple):
            loaded = loaded[0]
        self.model.load_state_dict(loaded, strict=False)
        self._freeze_backbone()
        logger.info("Checkpoint loaded successfully")

        # ── 5. Embedding hook (captures hidden states before lm_head) ─────
        #   Verified shapes: logits → [B,T,V], hidden → [T,B,H]
        self._last_hidden: torch.Tensor | None = None
        self.model.output_layer.register_forward_pre_hook(
            lambda _mod, args: setattr(self, "_last_hidden", args[0])
        )

    def _freeze_backbone(self) -> None:
        """Evo2 is used as a frozen feature extractor; only external heads train."""
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    # ------------------------------------------------------------------
    # Low-level forward
    # ------------------------------------------------------------------

    def _forward(self, sequences: Sequence[str], require_grad: bool = False):
        """Batched forward.  Returns logits [B,T,V], embeddings [B,T,H], mask [B,T], token_ids."""
        token_ids = [self.tokenizer.text_to_ids(s)[:self.max_length] for s in sequences]
        T = max(len(ids) for ids in token_ids)
        B = len(sequences)

        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        for i, ids in enumerate(token_ids):
            n = len(ids)
            input_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            mask[i, :n] = True
        position_ids = torch.arange(T, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)

        grad_context = torch.enable_grad() if require_grad else torch.no_grad()
        with grad_context:
            self._last_hidden = None
            logits = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=None).float()
            if not require_grad:
                logits.nan_to_num_(nan=0.0, posinf=80.0, neginf=-80.0)

            embeddings = None
            if self._last_hidden is not None:
                embeddings = self._last_hidden.permute(1, 0, 2).float()  # [T,B,H] → [B,T,H]
                if not require_grad:
                    embeddings.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        return logits, embeddings, mask, token_ids

    def _score(self, logits, input_ids_padded, mask):
        """Shifted autoregressive token probs → [B,T] clamped to [0,1]."""
        B, T, V = logits.shape
        log_p = F.log_softmax(logits, dim=-1)
        lp = torch.full((B, T), -float(np.log(V)), device=logits.device)
        lp[:, 1:] = log_p[:, :-1].gather(2, input_ids_padded[:, 1:].unsqueeze(-1)).squeeze(-1)
        return lp.exp().clamp(0.0, 1.0) * mask.float()

    # ------------------------------------------------------------------
    # GFM-Bench API
    # ------------------------------------------------------------------

    def infer_sequence_to_sequence(
        self, sequences: Sequence[str], conditional_input: Any = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        sequences = [str(s) for s in sequences]
        logits, embeddings, mask, token_ids = self._forward(sequences)
        B, T, V = logits.shape

        input_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
        for i, ids in enumerate(token_ids):
            input_ids[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=self.device)

        probs = self._score(logits, input_ids, mask)

        rep = None
        if embeddings is not None:
            m = mask.float().unsqueeze(-1)
            rep = ((embeddings * m).sum(1) / m.sum(1)).cpu().numpy()

        max_len = max(len(ids) for ids in token_ids)
        probs_np = np.zeros((B, max_len), dtype=np.float32)
        embs_np = np.zeros((B, max_len, self.hidden_size), dtype=np.float32) if embeddings is not None else None
        probs_cpu = probs.cpu().numpy()
        embs_cpu = embeddings.cpu().numpy() if embeddings is not None else None
        for i, ids in enumerate(token_ids):
            n = len(ids)
            probs_np[i, :n] = probs_cpu[i, :n]
            if embs_cpu is not None:
                embs_np[i, :n] = embs_cpu[i, :n]

        np.nan_to_num(probs_np, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        if embs_np is not None:
            np.nan_to_num(embs_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if rep is not None:
            np.nan_to_num(rep, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if embs_np is None:
            raise ValueError("Embeddings are None")
        if rep is None:
            raise ValueError("Representative is None")
        return probs_np, embs_np, rep

    def _sequence_to_representative(self, sequences: Sequence[str]) -> torch.Tensor:
        """Return representative embeddings as torch tensors for projection training."""
        sequences = [str(s) for s in sequences]
        require_grad = self.model.training and torch.is_grad_enabled()
        _, embeddings, mask, _ = self._forward(sequences, require_grad=require_grad)
        if embeddings is None:
            raise ValueError("Embeddings are None")

        m = mask.float().unsqueeze(-1)
        return (embeddings * m).sum(1) / m.sum(1).clamp_min(1.0)

    def infer_sequence_to_labels_probs(
        self,
        sequences: Sequence[str],
        conditional_input: Any = None,
    ) -> np.ndarray | None:
        """Direct sequence classification is not implemented for base Evo2."""
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self, variant_sequences: Sequence[str], ref_sequences: Sequence[str],
        conditional_input: Any = None,
    ) -> np.ndarray | None:
        var_p = self.infer_sequence_to_sequence(list(variant_sequences))[0]
        ref_p = self.infer_sequence_to_sequence(list(ref_sequences))[0]
        if var_p is None or ref_p is None:
            return None
        eps = 1e-8
        var_ll = np.log(np.clip(var_p, eps, 1.0)).sum(axis=1)
        ref_ll = np.log(np.clip(ref_p, eps, 1.0)).sum(axis=1)
        llr = np.clip(var_ll - ref_ll, -80, 80).astype(np.float32)
        p1 = 1.0 / (1.0 + np.exp(-llr))
        return np.stack([1.0 - p1, p1], axis=1).astype(np.float32)

    def sequence_pos_to_prob_pos(self, sequences: Sequence[str], pos: int) -> np.ndarray:
        return np.array([pos] * len(sequences))


    def infer_masked_sequence_to_token_probs(self, sequences: Sequence[str], variant_pos: int, variant_letters: Sequence[str], reference_letters: Sequence[str], conditional_input: Any = None) -> tuple[np.ndarray, np.ndarray]:
        return None, None

    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        """Keep the Evo2 backbone frozen/eval even during projection training."""
        self._freeze_backbone()
        return self

    def to(self, device: str):
        """Move the model to a CUDA device."""
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Evo2BioNeMoModel requires a CUDA device.")
        self.model.to(self.device)
        return self

    def parameters(self):
        """Return model parameters for the fine-tuner."""
        return self.model.parameters()

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return named model parameters."""
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def get_hidden_dim(self):
        """Return the hidden dimension of the model."""
        return self.hidden_dim

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def load_checkpoint(self, checkpoint_path: str):
        """Load Evo2 weights from a NeMo2 checkpoint directory/resource or a torch state dict."""
        ckpt_root = Path(checkpoint_path)
        if ckpt_root.is_file():
            state = torch.load(ckpt_root, map_location=self.device)
            state = state.get("model_state_dict", state)
            self.model.load_state_dict(state, strict=False)
            self._freeze_backbone()
            logger.info("Loaded torch checkpoint: %s", checkpoint_path)
            return self

        if not ckpt_root.is_dir():
            from bionemo.core.data.load import load
            ckpt_root = load(checkpoint_path)

        weights_root = ckpt_root / "weights" if (ckpt_root / "weights").is_dir() else ckpt_root
        sd = self.model.sharded_state_dict(prefix="module.")
        loaded = dist_ckpt_load(
            sd, str(weights_root),
            sharded_strategy=TorchDistLoadShardedStrategy(),
        )
        if isinstance(loaded, tuple):
            loaded = loaded[0]
        self.model.load_state_dict(loaded, strict=False)
        self._freeze_backbone()
        logger.info("Loaded Evo2 checkpoint: %s", checkpoint_path)
        return self