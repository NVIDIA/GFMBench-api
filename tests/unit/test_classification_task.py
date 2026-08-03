import numpy as np
import torch

from gfmbench_api import BaseGFMSupervisedClassificationTask as RootClassificationTask
from gfmbench_api.tasks import (
    BaseGFMSupervisedClassificationTask as TasksClassificationTask,
)
from gfmbench_api.tasks.base import BaseGFMSupervisedClassificationTask


class _ClassificationTask(BaseGFMSupervisedClassificationTask):
    def __init__(self, classification_mode, input_structure):
        self.classification_mode = classification_mode
        self.input_structure = input_structure
        super().__init__(".")

    def _get_default_max_seq_len(self):
        return 4

    def _get_num_classes(self):
        return 2

    def _get_num_labels(self):
        return 2 if self.classification_mode == "multi_label" else 1

    def _create_datasets(self):
        if self.classification_mode == "multi_label":
            labels = [
                torch.tensor([0.0, 1.0]),
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([1.0, 0.0]),
            ]
        else:
            labels = [0, 1, 0, 1]

        if self.input_structure == "variant_reference_pair":
            examples = [
                ("ACGT", "TGCA", label, torch.empty(0)) for label in labels
            ]
        else:
            examples = [("ACGT", label, torch.empty(0)) for label in labels]
        return examples, None, examples

    def get_task_name(self):
        return "test_classification"

    def get_conditional_input_meta_data_frame(self):
        return None


class _PerfectClassificationModel:
    @staticmethod
    def _single(batch_size):
        return np.array([[0.9, 0.1], [0.1, 0.9]] * (batch_size // 2))

    @staticmethod
    def _multi(batch_size):
        return np.array([[0.1, 0.9], [0.9, 0.1]] * (batch_size // 2))

    def infer_sequence_to_labels_probs(self, sequences, conditional_input=None):
        return self._single(len(sequences))

    def infer_variant_ref_sequences_to_labels_probs(
        self, variant_sequences, ref_sequences, conditional_input=None
    ):
        return self._single(len(variant_sequences))

    def infer_sequence_to_multilabel_probs(self, sequences, conditional_input=None):
        return self._multi(len(sequences))

    def infer_variant_ref_sequences_to_multilabel_probs(
        self, variant_sequences, ref_sequences, conditional_input=None
    ):
        return self._multi(len(variant_sequences))


def test_classification_attributes_have_unambiguous_counts():
    single = _ClassificationTask("single_label", "sequence").get_task_attributes()
    multi = _ClassificationTask("multi_label", "sequence").get_task_attributes()

    assert single["task_type"] == "classification"
    assert single["num_labels"] == 1
    assert single["num_classes"] == 2
    assert multi["num_labels"] == 2
    assert multi["num_classes"] == 2


def test_classification_base_does_not_supply_attribute_defaults():
    assert "classification_mode" not in BaseGFMSupervisedClassificationTask.__dict__
    assert "input_structure" not in BaseGFMSupervisedClassificationTask.__dict__


def test_classification_evaluation_supports_current_modes_and_inputs():
    model = _PerfectClassificationModel()
    for mode, structure in (
        ("single_label", "sequence"),
        ("single_label", "variant_reference_pair"),
        ("multi_label", "sequence"),
        ("multi_label", "variant_reference_pair"),
    ):
        scores = _ClassificationTask(mode, structure).eval_test_set(model)
        assert all(score == 1.0 for score in scores.values())


def test_classification_base_is_exported_from_public_packages():
    assert RootClassificationTask is BaseGFMSupervisedClassificationTask
    assert TasksClassificationTask is BaseGFMSupervisedClassificationTask
