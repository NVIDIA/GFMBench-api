import numpy as np
import pytest
import torch

from gfmbench_api import BaseGFMSupervisedRegressionTask as RootRegressionTask
from gfmbench_api.tasks import BaseGFMSupervisedRegressionTask as TasksRegressionTask
from gfmbench_api.tasks.base import BaseGFMSupervisedRegressionTask


class _RegressionTask(BaseGFMSupervisedRegressionTask):
    def __init__(self, output_spatiality):
        self.output_spatiality = output_spatiality
        super().__init__(".")

    def _get_default_max_seq_len(self):
        return 4

    def _get_num_outputs(self):
        return 2

    def _create_datasets(self):
        target_shape = (2,) if self.output_spatiality == "sequence" else (2, 2)
        examples = [
            ("ACGT", torch.full(target_shape, float(index)), torch.empty(0))
            for index in range(4)
        ]
        return examples, None, examples

    def get_task_name(self):
        return "test_regression"

    def get_conditional_input_meta_data_frame(self):
        return None


class _PerfectRegressionModel:
    def __init__(self, output_spatiality):
        self.output_spatiality = output_spatiality

    def infer_sequence_to_regression(self, sequences, conditional_input=None):
        values = np.arange(len(sequences), dtype=np.float32)
        if self.output_spatiality == "sequence":
            return np.repeat(values[:, None], 2, axis=1)
        return np.repeat(values[:, None, None], 4, axis=1).reshape(-1, 2, 2)


def test_regression_attributes_separate_objective_and_spatiality():
    task = _RegressionTask("binned")

    assert task.get_task_attributes()["task_type"] == "regression"
    assert task.get_task_attributes()["output_spatiality"] == "binned"
    assert task.get_task_attributes()["num_outputs"] == 2


def test_regression_base_is_exported_from_public_packages():
    assert RootRegressionTask is BaseGFMSupervisedRegressionTask
    assert TasksRegressionTask is BaseGFMSupervisedRegressionTask


def test_regression_rejects_unknown_spatiality():
    with pytest.raises(ValueError, match="output_spatiality"):
        _RegressionTask("unknown").get_task_attributes()


def test_regression_evaluation_supports_sequence_and_binned_outputs():
    for spatiality in ("sequence", "binned"):
        task = _RegressionTask(spatiality)
        scores = task.eval_test_set(_PerfectRegressionModel(spatiality))

        assert scores["regression_pearsonr_macro"] == 1.0
        assert scores["regression_r2_macro"] == 1.0
