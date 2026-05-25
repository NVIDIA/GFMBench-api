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
#
# This module does not embed third-party data download URLs.
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import (
    MultiLabelClassificationAccuracy,
    MultiLabelClassificationAUPRC,
    MultiLabelClassificationAUROC,
    MultiLabelClassificationMCC,
)
from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import (
    BaseGFMSupervisedSingleSeqTask,
)


class BaseGFMSupervisedSingleSeqLinearProbeTask(BaseGFMSupervisedSingleSeqTask):
    """
    Base class for single-sequence tasks evaluated with linear probing.

    Workflow:
    1. Run model forward pass on train/validation/test to extract sequence embeddings.
    2. Train logistic-regression probes on train embeddings.
    3. Select the best probe by validation accuracy.
    4. Evaluate on requested split with the selected probe (no probe re-training on eval split).
    """

    def __init__(self, root_data_dir_path: str, task_config: Optional[Dict[str, Any]] = None):
        self._selected_probe: Optional[LogisticRegression] = None
        self._probe_model_obj_id: Optional[int] = None
        self._probe_c_grid = self._parse_probe_c_grid(task_config or {})
        self._probe_max_iter = int((task_config or {}).get("logistic_regression_max_iter", 5000))
        self._probe_solver = str((task_config or {}).get("logistic_regression_solver", "lbfgs"))
        self._probe_random_state = int((task_config or {}).get("random_state", 42))
        class_weight = (task_config or {}).get("logistic_regression_class_weight", None)
        self._probe_class_weight = class_weight if class_weight in {None, "balanced"} else None
        super().__init__(root_data_dir_path=root_data_dir_path, task_config=task_config)

    @staticmethod
    def _parse_probe_c_grid(task_config: Dict[str, Any]) -> Tuple[float, ...]:
        raw = task_config.get("logistic_regression_c_grid", None)
        if raw is None:
            return (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
        if isinstance(raw, (int, float)):
            return (float(raw),)
        return tuple(float(v) for v in raw)

    @staticmethod
    def _labels_to_numpy(labels: Any) -> np.ndarray:
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy()
        return np.asarray(labels)

    @staticmethod
    def _conditional_to_numpy(conditional_input: Any) -> Optional[np.ndarray]:
        if conditional_input is None:
            return None
        if isinstance(conditional_input, torch.Tensor):
            return conditional_input.detach().cpu().numpy()
        return np.asarray(conditional_input)

    def _extract_embeddings_and_labels(
        self, model: Any, dataset: Dataset, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_embeddings = []
        all_labels = []

        for batch in tqdm(data_loader, desc=f"Embedding {split_name}"):
            sequences, labels, conditional_input = batch
            conditional_np = self._conditional_to_numpy(conditional_input)
            _, _, representative = self._safe_model_call(
                model,
                "infer_sequence_to_sequence",
                sequences,
                conditional_np,
                num_outputs=3,
            )
            if representative is None:
                raise ValueError(
                    "Model must implement infer_sequence_to_sequence and return sequence_representative "
                    "for linear-probe tasks."
                )
            all_embeddings.append(np.asarray(representative))
            all_labels.append(self._labels_to_numpy(labels))

        if len(all_embeddings) == 0:
            raise ValueError(f"No samples found while extracting {split_name} embeddings.")

        return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

    def _fit_probe_once(self, model: Any) -> None:
        if self.train_dataset is None:
            raise ValueError("Linear probing requires a training dataset, but train_dataset is None.")
        if self.validation_dataset is None:
            raise ValueError("Linear probing requires a validation dataset for model selection.")

        x_train, y_train = self._extract_embeddings_and_labels(model, self.train_dataset, "train")
        x_val, y_val = self._extract_embeddings_and_labels(model, self.validation_dataset, "validation")

        best_probe = None
        best_val_acc = -np.inf
        for c_value in self._probe_c_grid:
            probe = LogisticRegression(
                C=float(c_value),
                max_iter=self._probe_max_iter,
                solver=self._probe_solver,
                random_state=self._probe_random_state,
                class_weight=self._probe_class_weight,
            )
            probe.fit(x_train, y_train)
            val_preds = probe.predict(x_val)
            val_acc = float(accuracy_score(y_val, val_preds))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_probe = probe

        if best_probe is None:
            raise RuntimeError("Failed to train logistic-regression probe.")
        self._selected_probe = best_probe
        self._probe_model_obj_id = id(model)

    def _ensure_probe(self, model: Any) -> None:
        if self._selected_probe is None or self._probe_model_obj_id != id(model):
            self._fit_probe_once(model)

    def _predict_dataset_probs(
        self, model: Any, dataset: Dataset, split_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_probe(model)
        assert self._selected_probe is not None

        embeddings, labels = self._extract_embeddings_and_labels(model, dataset, split_name)
        raw_probs = self._selected_probe.predict_proba(embeddings)
        class_ids = self._selected_probe.classes_.astype(int)
        num_labels = self._get_num_labels()

        # Expand/reorder probabilities to [batch_size, num_labels].
        probs = np.zeros((raw_probs.shape[0], num_labels), dtype=np.float32)
        for idx, class_id in enumerate(class_ids):
            if 0 <= class_id < num_labels:
                probs[:, class_id] = raw_probs[:, idx]
        return probs, labels

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        probs, labels_np = self._predict_dataset_probs(model, dataset, "evaluation")

        metrics = [
            MultiLabelClassificationAccuracy(),
            MultiLabelClassificationMCC(),
            MultiLabelClassificationAUROC(),
            MultiLabelClassificationAUPRC(),
        ]
        for metric in metrics:
            metric.calc(probs, labels_np)

        results: Dict[str, Optional[float]] = {}
        for metric in metrics:
            results[metric.name] = metric.get_final_results()
        return results
