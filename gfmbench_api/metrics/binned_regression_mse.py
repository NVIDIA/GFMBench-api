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


class BinnedRegressionMSE(BaseMetric):
    """Mean squared error for binned-regression tasks.

    Accumulates squared error and element count across batches so the final
    value is the exact dataset-level MSE over all (sample, bin, track) entries.
    Computed in whatever target space the task uses (CAGE targets are log1p).
    """

    def reset(self):
        super().reset()
        self._sq_error_sum = 0.0
        self._count = 0

    @property
    def name(self):
        return "regression_mse"

    def _calc_impl(self, preds, targets):
        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        diff = preds - targets
        self._sq_error_sum += float((diff * diff).sum())
        self._count += diff.size

    def get_final_results(self):
        if self._count == 0:
            return None
        return self._sq_error_sum / self._count
