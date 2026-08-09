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


class ClassificationAUROC(BaseMetric):
    """
    Classification Area Under the Receiver Operating Characteristic curve (AUROC).

    Supports binary and multi-class single-label classification, as well as
    independent binary targets for multi-label classification.
    """

    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        super().__init__()

    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._probs_list = []
        self._gt_list = []

    @property
    def name(self):
        """Return the key name for results dictionary."""
        if self.multilabel:
            return "multilabel_auroc_macro"
        return "classification_auroc"

    def _calc_impl(self, probs, gt):
        """Store probabilities and labels for AUROC calculation."""
        self._probs_list.append(np.asarray(probs))
        self._gt_list.append(np.asarray(gt))

    def get_final_results(self):
        """Calculate AUROC from stored probabilities."""
        if not self._probs_list:
            return None

        # Concatenate all batches
        probs = np.concatenate(self._probs_list, axis=0)
        gt = np.concatenate(self._gt_list, axis=0)

        if self.multilabel:
            if probs.ndim != 2:
                return None

            scores = []
            for label_idx in range(probs.shape[1]):
                y_true = gt[:, label_idx]
                if np.unique(y_true).size < 2:
                    continue
                scores.append(roc_auc_score(y_true, probs[:, label_idx]))
            return float(np.mean(scores)) if scores else None

        if probs.shape[1] == 2:
            # Binary classification: use probabilities of positive class
            return roc_auc_score(gt, probs[:, 1])
        # Multi-class: use macro averaging with ovr (one-vs-rest)
        return roc_auc_score(gt, probs, multi_class="ovr", average="macro")
