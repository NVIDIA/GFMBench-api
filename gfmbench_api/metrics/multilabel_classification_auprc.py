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
from sklearn.metrics import average_precision_score
from .base_metric import BaseMetric


class MultiLabelClassificationAUPRC(BaseMetric):
    """
    Macro AUPRC (average precision) over independent binary labels.

    Receives per-label sigmoid probabilities of shape [batch, num_labels] and
    binary ground-truth labels of the same shape. Average precision is computed
    per label and averaged over labels that contain at least one positive.
    """

    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._probs_list = []
        self._gt_list = []

    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "multilabel_auprc_macro"

    def _calc_impl(self, probs, gt):
        """Store probabilities and labels for AUPRC calculation."""
        self._probs_list.append(np.asarray(probs))
        self._gt_list.append(np.asarray(gt))

    def get_final_results(self):
        """Calculate the mean per-label average precision from stored probabilities."""
        if not self._probs_list:
            return None

        # Concatenate all batches
        probs = np.concatenate(self._probs_list, axis=0)
        gt = np.concatenate(self._gt_list, axis=0)
        if probs.ndim != 2:
            return None

        scores = []
        for label_idx in range(probs.shape[1]):
            y_true = gt[:, label_idx]
            # Average precision needs at least one positive to be meaningful.
            if y_true.sum() == 0:
                continue
            scores.append(average_precision_score(y_true, probs[:, label_idx]))

        return float(np.mean(scores)) if scores else None

