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
import numpy as np
from sklearn.metrics import roc_auc_score
from .base_metric import BaseMetric


class MultiLabelClassificationAUROC(BaseMetric):
    """Macro AUROC over independent binary labels."""

    def reset(self):
        super().reset()
        self._probs_list = []
        self._gt_list = []

    @property
    def name(self):
        return "multilabel_auroc_macro"

    def _calc_impl(self, probs, gt):
        self._probs_list.append(np.asarray(probs))
        self._gt_list.append(np.asarray(gt))

    def get_final_results(self):
        if not self._probs_list:
            return None
        probs = np.concatenate(self._probs_list, axis=0)
        gt = np.concatenate(self._gt_list, axis=0)
        if probs.ndim != 2:
            return None

        scores = []
        for label_idx in range(probs.shape[1]):
            y_true = gt[:, label_idx]
            if np.unique(y_true).size < 2:
                continue
            scores.append(roc_auc_score(y_true, probs[:, label_idx]))

        return float(np.mean(scores)) if scores else None

