# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact Evo2 task outputs preserve scores without full CPU hidden arrays."""

import ast
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.e2e.fixture_tasks import FixtureZeroShotIndelTask, FixtureZeroShotSNVTask
from usage_examples.run_benchmark import parse_args, select_compact_outputs
from usage_examples.trainers.model_wrapper import GFMWithProjection


class SequenceModel:
    def __init__(self):
        self.full_calls = 0
        self.compact_calls = 0

    def _outputs(self, sequences):
        length = max(map(len, sequences))
        probs = np.full((len(sequences), length), 0.25, dtype=np.float32)
        hidden = np.zeros((len(sequences), length, 4), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            for position, letter in enumerate(sequence):
                hidden[row, position] = [ord(letter) / 100, position / 100, row / 100, 1]
        return probs, hidden, hidden.mean(axis=1)

    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        self.full_calls += 1
        return self._outputs(sequences)

    def sequence_pos_to_prob_pos(self, sequences, position):
        return np.full(len(sequences), position, dtype=np.int32)

    def infer_masked_sequence_to_token_probs(self, sequences, position, variants, references):
        return np.full(len(sequences), 0.25), np.full(len(sequences), 0.25)


class CompactSequenceModel(SequenceModel):
    def infer_sequence_to_metrics(self, sequences, conditional_input=None, position=None):
        self.compact_calls += 1
        probs, hidden, representative = self._outputs(sequences)
        selected = None if position is None else hidden[:, position:position + 1]
        return probs, selected, representative


def test_benchmark_selects_compact_for_supported_tasks_by_default():
    base = ['--csv_path', '/tmp/report.csv', '--root_data_dir_path', '/tmp/data',
            '--model', 'Evo2', '--linear_prob']
    class SupportedModel:
        def infer_sequence_to_metrics(self, sequences, conditional_input=None, position=None):
            return None

        def infer_representative(self, sequences, conditional_input=None):
            return None

    for flags, expected in [([], True), (['--compact_outputs'], True),
                            (['--full_outputs'], False)]:
        args = parse_args(base + flags)
        for supervised in [False, True]:
            model = SupportedModel()
            assert select_compact_outputs(
                model, supervised=supervised, requested=args.compact_outputs,
            ) is expected
            assert model.compact_outputs is expected
    with pytest.raises(SystemExit):
        parse_args(base + ['--compact_outputs', '--full_outputs'])

    class FullOnlyModel:
        pass

    model = FullOnlyModel()
    assert select_compact_outputs(model, supervised=False, requested=None) is False
    with pytest.raises(ValueError, match='infer_sequence_to_metrics'):
        select_compact_outputs(model, supervised=False, requested=True)


@pytest.mark.parametrize('task_type', [FixtureZeroShotSNVTask, FixtureZeroShotIndelTask])
def test_zero_shot_compact_matches_full_outputs(fixtures_root, smoke_task_config, task_type):
    task = task_type(str(fixtures_root), smoke_task_config)
    original = SequenceModel()
    compact = CompactSequenceModel()
    select_compact_outputs(compact, supervised=False, requested=None)

    expected = task.eval_test_set(original)
    actual = task.eval_test_set(compact)

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value), name
    assert compact.compact_calls > 0
    assert compact.full_calls == 0

    disabled = CompactSequenceModel()
    select_compact_outputs(disabled, supervised=False, requested=False)
    task.eval_test_set(disabled)
    assert disabled.full_calls > 0
    assert disabled.compact_calls == 0


def test_projection_uses_compact_representative_without_full_outputs():
    class RepresentativeModel:
        device = 'cpu'

        def infer_representative(self, sequences, conditional_input=None):
            return torch.tensor([[len(sequence), 1.] for sequence in sequences])

        def infer_sequence_to_sequence(self, sequences, conditional_input=None):
            raise AssertionError('The full sequence output is unnecessary')

    projection = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2))
    model = RepresentativeModel()
    select_compact_outputs(model, supervised=True, requested=None)
    wrapped = GFMWithProjection(model, projection, cache_size=0.01)

    single = wrapped.infer_sequence_to_labels_probs(['AC', 'ACG'])
    pair_projection = torch.nn.Linear(4, 2, bias=False)
    paired_wrapper = GFMWithProjection(model, pair_projection, cache_size=0.01)
    paired = paired_wrapper.infer_variant_ref_sequences_to_labels_probs(['AC', 'ACG'], ['AC', 'AC'])
    unprojected = GFMWithProjection(model).infer_sequence_to_labels_probs(['AC'])

    assert single.shape == (2, 2)
    assert paired.shape == (2, 2)
    assert isinstance(unprojected, np.ndarray)
    np.testing.assert_allclose(single, torch.softmax(torch.tensor([[2., 1.], [3., 1.]]), 1).numpy())


def test_evo2_compact_outputs_match_its_full_cpu_path():
    # Import the real method bodies without requiring Megatron on a CPU test host.
    path = Path(__file__).resolve().parents[2] / 'usage_examples/sanity_models/evo2_model.py'
    tree = ast.parse(path.read_text())
    model_class = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'Evo2BioNeMoModel')
    names = {'_forward', '_score', 'infer_sequence_to_sequence',
             'infer_sequence_to_metrics', 'sequence_pos_to_prob_pos'}
    methods = [node for node in model_class.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    selected_class = ast.ClassDef(name='CpuEvo2', bases=[], keywords=[], body=methods, decorator_list=[])
    selected_module = ast.fix_missing_locations(ast.Module(body=[selected_class], type_ignores=[]))
    namespace = dict(torch=torch, np=np, F=F, Any=Any, Sequence=Sequence)
    exec(compile(selected_module, str(path), 'exec'), namespace)

    class Tokenizer:
        def text_to_ids(self, sequence):
            return list(sequence.encode('ascii'))

    model = namespace['CpuEvo2']()
    model.device, model.max_length, model.hidden_size = torch.device('cpu'), 64, 8
    model.tokenizer = Tokenizer()

    def forward(input_ids, position_ids, attention_mask):
        values = input_ids.float()
        hidden = (values[:, :, None] / 97 + torch.arange(8) / 11).bfloat16()
        model._last_hidden = hidden.permute(1, 0, 2)
        return (values[:, :, None] / 97 + torch.arange(128) / 13).bfloat16()

    model.model = forward
    sequences = ['ACGT', 'AC', 'TCAA']
    full_probs, full_hidden, full_representative = model.infer_sequence_to_sequence(sequences)
    probs, selected, representative = model.infer_sequence_to_metrics(sequences, position=1)

    np.testing.assert_array_equal(probs, full_probs)
    np.testing.assert_array_equal(representative, full_representative)
    np.testing.assert_array_equal(selected[:, 0], full_hidden[:, 1])
    assert selected.nbytes < full_hidden.nbytes
