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
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from gfmbench_api.metrics import MultiTrackBinaryAUPRC, MultiTrackBinaryAUROC
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class BaseGFMSupervisedMultiLabelTask(BaseGFMTask):
    """Base class for multi-label classification (independent binary tracks).

    This differs from ``BaseGFMSupervisedMultiClassTask`` (single-label
    softmax) in three ways, which is why it is a sibling base rather than a
    subclass:

    * **Labels** are a per-track binary vector, so each example is
      ``(sequence, label_vector[float32, num_labels], conditional_input)``.
    * **Training** uses binary-cross-entropy-with-logits, not cross-entropy.
      ``get_task_attributes`` advertises ``task_type='multilabel'`` so the
      fine-tuner picks the BCE path.
    * **Eval** uses independent sigmoid probabilities (they do not sum to 1)
      and macro AUROC / AUPRC averaged over tracks.

    Subclasses must implement ``_get_num_labels``, ``_create_datasets``,
    ``get_task_name``, ``_get_default_max_seq_len`` and
    ``get_conditional_input_meta_data_frame``.
    """

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for multi-label classification tasks."""
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "is_variant_effect_prediction": False,
            "num_labels": self._get_num_labels(),
            "task_type": "multilabel",
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Evaluate per-track sigmoid probabilities with macro AUROC/AUPRC."""
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        metrics = [MultiTrackBinaryAUROC(), MultiTrackBinaryAUPRC()]

        for batch in tqdm(data_loader, desc="Evaluating"):
            sequences, labels, conditional_input = batch

            # Shape: [batch_size, num_labels] of independent sigmoid probs.
            probs, = self._safe_model_call(
                model, "infer_sequence_to_multilabel_probs", sequences, conditional_input, num_outputs=1
            )

            labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

            if probs is not None:
                num_labels = self._get_num_labels()
                assert probs.shape[1] == num_labels, \
                    f"Expected {num_labels} tracks, but got {probs.shape[1]}"

            for metric in metrics:
                metric.calc(probs, labels_np)

        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_num_labels(self) -> int:
        """Subclasses must implement this: return number of independent tracks."""
        pass
