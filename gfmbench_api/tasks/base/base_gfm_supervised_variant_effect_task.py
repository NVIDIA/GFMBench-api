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

from typing import Any, Optional, Tuple

import numpy as np

from gfmbench_api.tasks.base.base_gfm_supervised_classification_task import (
    BaseGFMSupervisedClassificationTask,
    ClassificationMode,
    InputStructure,
)


class BaseGFMSupervisedVariantEffectTask(BaseGFMSupervisedClassificationTask):
    """Base class for supervised classification from variant/reference pairs."""

    def _get_input_structure(self) -> InputStructure:
        return InputStructure.VARIANT_REFERENCE_PAIR

    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        variant_sequences, ref_sequences, labels, conditional_input = batch
        method = (
            "infer_variant_ref_sequences_to_multilabel_probs"
            if self._get_classification_mode() == ClassificationMode.MULTI_LABEL
            else "infer_variant_ref_sequences_to_labels_probs"
        )
        probs, = self._safe_model_call(
            model,
            method,
            variant_sequences,
            ref_sequences,
            conditional_input,
            num_outputs=1,
        )
        return probs, labels
