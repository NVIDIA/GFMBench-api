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

from .base_metric import BaseMetric


class BinnedRegressionR2(BaseMetric):
    """Macro coefficient of determination (R^2) over tracks for binned regression.

    Receives predictions and targets of shape [batch, num_bins, num_tracks]
    (binned tracks, e.g. CAGE). All (sample, bin) pairs are pooled per track,
    R^2 = 1 - SS_res / SS_tot is computed for each track, and the mean over
    tracks is returned. This is distinct from the (squared) Pearson correlation
    reported by ``BinnedRegressionPearsonR``.

    Tracks whose targets are constant (``SS_tot == 0``) have an undefined R^2 and
    are skipped in the macro average.
    """

    def reset(self):
        super().reset()
        self._pred_list = []
        self._target_list = []

    @property
    def name(self):
        return "regression_r2_macro"

    def _calc_impl(self, preds, targets):
        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        # Collapse leading dims (batch, bins) -> rows, keep tracks as columns.
        num_tracks = preds.shape[-1]
        self._pred_list.append(preds.reshape(-1, num_tracks))
        self._target_list.append(targets.reshape(-1, num_tracks))

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        if ss_tot <= 0:
            return np.nan
        ss_res = float(((y_true - y_pred) ** 2).sum())
        return 1.0 - ss_res / ss_tot

    def get_final_results(self):
        if not self._pred_list:
            return None
        preds = np.concatenate(self._pred_list, axis=0)
        targets = np.concatenate(self._target_list, axis=0)

        scores = []
        for track in range(preds.shape[1]):
            r2 = self._r2(targets[:, track], preds[:, track])
            if not np.isnan(r2):
                scores.append(r2)

        return float(np.mean(scores)) if scores else None
