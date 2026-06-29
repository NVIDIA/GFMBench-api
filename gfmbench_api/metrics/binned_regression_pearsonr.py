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


class BinnedRegressionPearsonR(BaseMetric):
    """Macro Pearson correlation over tracks for binned-regression tasks.

    Receives predictions and targets of shape [batch, num_bins, num_tracks]
    (binned tracks, e.g. CAGE). All (sample, bin) pairs are pooled per track,
    Pearson r is computed for each track, and the mean over tracks is returned.
    This matches the Basenji/Enformer convention of reporting correlation across
    genomic positions per output track.
    """

    def reset(self):
        super().reset()
        self._pred_list = []
        self._target_list = []

    @property
    def name(self):
        return "regression_pearsonr_macro"

    def _calc_impl(self, preds, targets):
        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        # Collapse leading dims (batch, bins) -> rows, keep tracks as columns.
        num_tracks = preds.shape[-1]
        self._pred_list.append(preds.reshape(-1, num_tracks))
        self._target_list.append(targets.reshape(-1, num_tracks))

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        if denom <= 0:
            return np.nan
        return float((x * y).sum() / denom)

    def get_final_results(self):
        if not self._pred_list:
            return None
        preds = np.concatenate(self._pred_list, axis=0)
        targets = np.concatenate(self._target_list, axis=0)

        scores = []
        for track in range(preds.shape[1]):
            r = self._pearson(preds[:, track], targets[:, track])
            if not np.isnan(r):
                scores.append(r)

        return float(np.mean(scores)) if scores else None
