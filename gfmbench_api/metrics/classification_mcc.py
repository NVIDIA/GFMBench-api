# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .base_metric import BaseMetric


class ClassificationMCC(BaseMetric):
    """MCC for single-label binary or multiclass classification."""

    def reset(self):
        super().reset()
        self._predictions_list = []
        self._gt_list = []

    @property
    def name(self):
        return "classification_mcc"

    def _calc_impl(self, probs, gt):
        self._predictions_list.append(np.argmax(probs, axis=1))
        self._gt_list.append(gt)

    def get_final_results(self):
        if not self._predictions_list:
            return None
        predictions = np.concatenate(self._predictions_list)
        gt = np.concatenate(self._gt_list)
        return matthews_corrcoef(gt, predictions)
