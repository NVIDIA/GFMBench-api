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
from sklearn.metrics import matthews_corrcoef
from .base_metric import BaseMetric


class MultiLabelClassificationMCC(BaseMetric):
    """
    Multi-label classification Matthews Correlation Coefficient (MCC) metric.
    
    Receives probabilities per label for the entire dataset and ground truth labels.
    Performs argmax on probabilities and compares to ground truth for MCC score.
    """
    
    def __init__(self):
        """Initialize storage for probabilities and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._predictions_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "classification_mcc"
    
    def _calc_impl(self, probs, gt):
        """Compute predictions via argmax and store them."""
        # Perform argmax to get predicted labels
        predictions = np.argmax(probs, axis=1)
        
        # Store only predictions and labels
        self._predictions_list.append(predictions)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate MCC from stored predictions."""
        if not self._predictions_list:
            return None
        
        # Concatenate all batches
        all_predictions = np.concatenate(self._predictions_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate MCC
        return matthews_corrcoef(all_gt, all_predictions)

