# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel
from gfmbench_api.tasks.base.base_gfm_supervised_regression_task import (
    BaseGFMSupervisedRegressionTask,
)
from gfmbench_api.tasks.concrete import lrb_cage_task
from gfmbench_api.utils.binning import get_valid_token_counts, pool_tokens_to_bins


class RegressionTask(BaseGFMSupervisedRegressionTask):
    def __init__(self, bin_size_bp=None, max_sequence_length=512):
        self.bin_size_bp = bin_size_bp
        super().__init__("unused", {"max_sequence_length": max_sequence_length})

    def get_bin_size_bp(self):
        return self.bin_size_bp

    def _get_num_labels(self):
        return 2

    def _get_default_max_seq_len(self):
        return 512

    def get_task_name(self):
        return "test_regression"

    def get_conditional_input_meta_data_frame(self):
        return None

    def _create_datasets(self):
        return [], None, []


@pytest.mark.parametrize("bin_size_bp, expected", [(None, 1), (128, 4), (256, 2)])
def test_attributes_derive_bin_count(bin_size_bp, expected):
    attributes = RegressionTask(bin_size_bp).get_task_attributes()

    assert attributes["num_bins"] == expected
    assert attributes["num_labels"] == 2
    assert "output_spatiality" not in attributes


@pytest.mark.parametrize("bin_size_bp", [True, False, 0, -128, 128.0, "128"])
def test_invalid_bin_width_is_rejected(bin_size_bp):
    with pytest.raises(ValueError, match="positive int or None"):
        RegressionTask(bin_size_bp).get_task_attributes()


@pytest.mark.parametrize(
    "sequence_length, message",
    [(500, "must be a multiple"), (128, "at least 2 bins"), (0, "at least 2 bins")],
)
def test_invalid_binned_sequence_length_is_rejected(sequence_length, message):
    with pytest.raises(ValueError, match=message):
        RegressionTask(128, sequence_length).get_task_attributes()


class ModelWithHead:
    """Model returning final regression predictions via infer_sequence_to_regression_outputs.

    ``outputs`` maps a sequence to its prediction: [num_labels] for sequence-level
    tasks, [num_bins, num_labels] for binned tasks.
    """

    def __init__(self, outputs=None):
        self.outputs = outputs

    def infer_sequence_to_regression_outputs(self, sequences, conditional_input=None):
        if self.outputs is None:
            return None
        return np.stack([self.outputs[s] for s in sequences])


def test_sequence_level_evaluation_scores_regression_outputs():
    task = RegressionTask(None)
    task.batch_size = 2
    targets = {
        base * 512: np.arange(2, dtype=np.float32) + i for i, base in enumerate("ACGT")
    }
    dataset = [(seq, label, np.array([])) for seq, label in targets.items()]

    scores = task._eval_dataset(ModelWithHead(targets), dataset)

    assert scores["regression_pearsonr_macro"] == pytest.approx(1.0)
    assert scores["regression_r2_macro"] == pytest.approx(1.0)


def test_binned_evaluation_scores_regression_outputs():
    task = RegressionTask(128)  # 512 bp -> 4 bins
    task.batch_size = 2
    targets = {
        base * 512: np.arange(8, dtype=np.float32).reshape(4, 2) + i
        for i, base in enumerate("ACGT")
    }
    dataset = [(seq, label, np.array([])) for seq, label in targets.items()]

    scores = task._eval_dataset(ModelWithHead(targets), dataset)

    assert scores["regression_pearsonr_macro"] == pytest.approx(1.0)
    assert scores["regression_r2_macro"] == pytest.approx(1.0)


@pytest.mark.parametrize("bin_size_bp", [None, 128])
def test_model_without_regression_outputs_yields_none_metrics(bin_size_bp):
    task = RegressionTask(bin_size_bp)
    shape = (2,) if bin_size_bp is None else (4, 2)
    dataset = [("A" * 512, np.zeros(shape), np.array([]))]
    expected = {"regression_pearsonr_macro": None, "regression_r2_macro": None}

    # Same convention as classification: a model that does not implement the method,
    # or returns None in the slots the task reads, is skipped silently.
    assert task._eval_dataset(object(), dataset) == expected
    assert task._eval_dataset(ModelWithHead(), dataset) == expected


@pytest.mark.parametrize("bin_size_bp", [None, 128])
def test_wrong_output_width_is_rejected(bin_size_bp):
    task = RegressionTask(bin_size_bp)
    shape = (2,) if bin_size_bp is None else (4, 2)
    dataset = [("A" * 512, np.zeros(shape), np.array([]))]
    model = ModelWithHead({"A" * 512: np.zeros(shape[:-1] + (3,))})

    with pytest.raises(ValueError, match="Expected 2 labels"):
        task._eval_dataset(model, dataset)


def test_wrapper_rejects_non_3d_token_embeddings_when_binned():
    from usage_examples.trainers import GFMWithProjection

    class Backbone:
        device = "cpu"

        def infer_sequence_to_sequence(self, sequences, conditional_input=None):
            return None, np.zeros((len(sequences), 4), np.float32), None

    wrapped = GFMWithProjection(
        Backbone(), torch.nn.Linear(4, 2), num_labels=2, task_type="regression", num_bins=4
    )

    with pytest.raises(ValueError, match="3 dimensions"):
        wrapped.infer_sequence_to_regression_outputs(["A" * 512])


def test_sequence_level_evaluation_rejects_wrong_ndim():
    task = RegressionTask(None)
    dataset = [("A" * 512, np.zeros(2), np.array([]))]
    model = SimpleNamespace(
        infer_sequence_to_regression_outputs=lambda seqs, ci=None: np.zeros((1, 1, 2)),
    )

    with pytest.raises(ValueError, match="2 dimensions"):
        task._eval_dataset(model, dataset)


def test_regression_inference_is_not_a_base_model_method():
    assert not hasattr(BaseGFMModel, "infer_sequence_to_regression")


class TestBinning:
    def test_fractional_pooling_matches_manual_bins(self):
        values = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        pooled = pool_tokens_to_bins(values, torch.tensor([6]), 3)
        expected = torch.stack([values[0, 0:2].mean(0), values[0, 2:4].mean(0), values[0, 4:6].mean(0)])
        assert torch.allclose(pooled[0], expected)

    def test_padding_tokens_are_ignored_and_empty_bins_are_zero(self):
        values = torch.ones(1, 8, 1)
        values[0, 4:] = 100.0  # padding beyond the 4 valid tokens
        pooled = pool_tokens_to_bins(values, torch.tensor([4]), 8)
        assert pooled[0, :, 0].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    def test_valid_token_counts_use_last_position_mapping(self):
        model = SimpleNamespace(
            sequence_pos_to_prob_pos=lambda seqs, pos: np.array(
                [min(pos, len(s) // 2) for s in seqs]
            )
        )
        counts = get_valid_token_counts(model, ["ACGT", "AC", "ACGTAC"], 6)
        assert counts.tolist() == [3, 2, 4]

    def test_unmappable_position_assumes_full_length(self):
        model = SimpleNamespace(sequence_pos_to_prob_pos=lambda seqs, pos: np.full(len(seqs), -1))
        assert get_valid_token_counts(model, ["ACGT"], 4).tolist() == [4]


@pytest.mark.parametrize(
    "configured_length, effective_length",
    [(256, 256), (384, 256), (2500, 2304), (114688, 114688), (200000, 114688)],
)
def test_cage_rounds_length_before_building_datasets(
    monkeypatch, tmp_path, configured_length, effective_length
):
    def build_examples(**kwargs):
        assert kwargs["sequence_length"] == effective_length
        label = np.zeros((effective_length // 128, 50), dtype=np.float32)
        return [], [], [("A" * effective_length, label, np.array([]))]

    monkeypatch.setattr(lrb_cage_task, "build_cage_examples", build_examples)
    task = lrb_cage_task.LRBCagePredictionTask(
        str(tmp_path), {"max_sequence_length": configured_length}
    )

    assert task.max_sequence_length == effective_length
    assert task.get_task_attributes()["num_bins"] == task.test_dataset[0][1].shape[0]


@pytest.mark.parametrize("configured_length", [0, 128, 255])
def test_cage_rejects_too_short_a_window_before_building(
    monkeypatch, tmp_path, configured_length
):
    def unexpected_build(**kwargs):
        pytest.fail("Invalid CAGE lengths must fail before loading data")

    monkeypatch.setattr(lrb_cage_task, "build_cage_examples", unexpected_build)
    with pytest.raises(ValueError, match="at least 2 bins"):
        lrb_cage_task.LRBCagePredictionTask(
            str(tmp_path), {"max_sequence_length": configured_length}
        )


def test_wrapper_and_finetuner_regression_round_trip():
    import torch.nn as nn
    from usage_examples.trainers import GFMFinetuner, GFMWithProjection

    hidden, length, bins = 4, 8, 2
    seen_conditional = []

    class Backbone:
        device = "cpu"

        def infer_sequence_to_sequence(self, sequences, conditional_input=None):
            seen_conditional.append(conditional_input)
            emb = np.stack(
                [np.full((length, hidden), float(len(s) + i), np.float32) for i, s in enumerate(sequences)]
            )
            return None, emb, emb.mean(axis=1)

        def sequence_pos_to_prob_pos(self, sequences, pos):
            return np.full(len(sequences), pos)

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    backbone = Backbone()
    head = nn.Linear(hidden, 2)
    wrapped = GFMWithProjection(backbone, head, task_type="regression", num_bins=bins)
    _, tokens, rep = wrapped.infer_sequence_to_sequence(["A" * length, "C" * length])
    assert tokens.shape == (2, length, hidden) and rep.shape == (2, hidden)

    # Linear head: head-then-pool (wrapper) equals pool-then-head.
    task = RegressionTask(4, max_sequence_length=length)
    preds = task._predict_batch(wrapped, ["A" * length, "C" * length], None, 2)
    assert preds.shape == (2, bins, 2)
    pooled_first = head(
        pool_tokens_to_bins(
            torch.from_numpy(backbone.infer_sequence_to_sequence(["A" * length, "C" * length])[1]),
            torch.tensor([length, length]),
            bins,
        )
    ).detach().numpy()
    np.testing.assert_allclose(preds, pooled_first, rtol=1e-5, atol=1e-5)

    conditional = torch.ones(2, 3)
    loader = [(["A" * length, "C" * length], torch.zeros(2, bins, 2), conditional)]
    seen_conditional.clear()
    tuner = GFMFinetuner(
        backbone, loader, hidden, 2, 1, 1e-2, only_proj_layer=True,
        num_labels=2, task_type="regression", num_bins=bins,
    )
    wrapped_trained = tuner.fine_tune()
    assert wrapped_trained.task_type == "regression" and wrapped_trained.projection_layer is not None
    assert seen_conditional and all(c is conditional for c in seen_conditional)
    with pytest.raises(ValueError, match="frozen backbone"):
        GFMFinetuner(backbone, loader, hidden, 2, 1, 1e-2, only_proj_layer=False, task_type="regression")
