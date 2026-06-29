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

from gfmbench_api.metrics import BinnedRegressionMSE, BinnedRegressionPearsonR
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class BaseGFMSupervisedBinnedRegressionTask(BaseGFMTask):
    """Base class for binned multi-track regression tasks (e.g. CAGE).

    Each example predicts a ``[num_bins, num_tracks]`` matrix of continuous
    values, where ``num_bins = sequence_length / bin_size_bp``. This needs a
    different training/eval path from classification, so it is a sibling base:

    * **Labels** are a 2D ``[num_bins, num_tracks]`` float array, so each example
      is ``(sequence, label_matrix[float32], conditional_input)``.
    * **Training** uses MSE over the binned predictions. ``get_task_attributes``
      advertises ``task_type='binned_regression'`` so the fine-tuner pools token
      embeddings into ``num_bins`` bins and trains a per-bin regression head.
    * **Eval** reports macro Pearson r (per track, across all bins/samples) and
      dataset-level MSE via ``infer_sequence_to_binned_tracks``.

    ``num_labels`` carries the number of output tracks so the shared fine-tuner
    can size the regression head as ``Linear(hidden_dim, num_tracks)``.

    Subclasses must implement ``_get_num_labels`` (number of tracks),
    ``_create_datasets``, ``get_task_name``, ``_get_default_max_seq_len`` and
    ``get_conditional_input_meta_data_frame``.
    """

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for binned-regression tasks."""
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "is_variant_effect_prediction": False,
            "num_labels": self._get_num_labels(),
            "task_type": "binned_regression",
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Evaluate binned predictions with macro Pearson r and MSE."""
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        metrics = [BinnedRegressionPearsonR(), BinnedRegressionMSE()]

        for batch in tqdm(data_loader, desc="Evaluating"):
            sequences, labels, conditional_input = batch

            # Shape: [batch_size, num_bins, num_tracks] of regression outputs.
            preds, = self._safe_model_call(
                model, "infer_sequence_to_binned_tracks", sequences, conditional_input, num_outputs=1
            )

            labels_np = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

            if preds is not None:
                num_tracks = self._get_num_labels()
                assert preds.shape[-1] == num_tracks, \
                    f"Expected {num_tracks} tracks, but got {preds.shape[-1]}"

            for metric in metrics:
                metric.calc(preds, labels_np)

        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def _get_num_labels(self) -> int:
        """Subclasses must implement this: return number of output tracks."""
        pass
