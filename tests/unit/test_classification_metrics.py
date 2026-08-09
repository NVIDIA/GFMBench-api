import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from gfmbench_api.metrics import ClassificationAUPRC, ClassificationAUROC


@pytest.mark.parametrize(
    ("metric", "expected_name"),
    [
        (ClassificationAUROC(), "classification_auroc"),
        (ClassificationAUPRC(), "classification_auprc"),
        (ClassificationAUROC(multilabel=True), "multilabel_auroc_macro"),
        (ClassificationAUPRC(multilabel=True), "multilabel_auprc_macro"),
    ],
)
def test_classification_metric_names(metric, expected_name):
    assert metric.name == expected_name


def test_single_label_metrics_preserve_multiclass_behavior():
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )
    gt = np.array([0, 1, 2, 0, 1, 2])

    auroc = ClassificationAUROC()
    auprc = ClassificationAUPRC()
    for metric in (auroc, auprc):
        metric.calc(probs[:3], gt[:3])
        metric.calc(probs[3:], gt[3:])

    assert auroc.get_final_results() == pytest.approx(
        roc_auc_score(gt, probs, multi_class="ovr", average="macro")
    )
    assert auprc.get_final_results() == pytest.approx(
        average_precision_score(gt, probs, average="macro")
    )


def test_multilabel_metrics_skip_undefined_labels():
    probs = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.3, 0.4],
            [0.2, 0.1, 0.8],
            [0.9, 0.4, 0.3],
        ]
    )
    gt = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )

    auroc = ClassificationAUROC(multilabel=True)
    auprc = ClassificationAUPRC(multilabel=True)
    for metric in (auroc, auprc):
        metric.calc(probs, gt)

    assert auroc.get_final_results() == pytest.approx(1.0)
    assert auprc.get_final_results() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    [ClassificationAUROC(multilabel=True), ClassificationAUPRC(multilabel=True)],
)
def test_multilabel_metrics_return_none_without_valid_labels(metric):
    metric.calc(np.full((3, 2), 0.25), np.zeros((3, 2), dtype=int))
    assert metric.get_final_results() is None
