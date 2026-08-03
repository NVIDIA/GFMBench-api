# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.metrics import roc_auc_score

from .base_metric import BaseMetric


class ClassificationAUROC(BaseMetric):
    """AUROC for single-label binary or multiclass classification."""

    def reset(self):
        super().reset()
        self._probs_list = []
        self._gt_list = []

    @property
    def name(self):
        return "classification_auroc"

    def _calc_impl(self, probs, gt):
        self._probs_list.append(probs)
        self._gt_list.append(gt)

    def get_final_results(self):
        if not self._probs_list:
            return None
        probs = np.concatenate(self._probs_list)
        gt = np.concatenate(self._gt_list)
        if probs.shape[1] == 2:
            return roc_auc_score(gt, probs[:, 1])
        return roc_auc_score(gt, probs, multi_class="ovr", average="macro")
