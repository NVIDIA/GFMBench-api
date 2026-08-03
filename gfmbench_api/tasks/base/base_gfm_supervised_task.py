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

from abc import abstractmethod
from typing import Optional

from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class BaseGFMSupervisedTask(BaseGFMTask):
    """
    Base class for supervised tasks, which are fine-tuned on training data.

    Supervised tasks expose their training data through get_finetune_dataset() and
    declare how many labels they predict through _get_num_labels(): the number of
    classification targets for classification tasks, or the number of continuous
    outputs for regression tasks.

    Subclasses must implement:
        - _get_num_labels(): Return number of labels predicted per sequence or bin
    """

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def _validate_num_labels(self) -> int:
        """Return the number of labels, verifying that the subclass declares at least one."""
        num_labels = self._get_num_labels()
        if num_labels < 1:
            raise ValueError(f"num_labels must be positive, got {num_labels}")
        return num_labels

    @abstractmethod
    def _get_num_labels(self) -> int:
        """Subclasses must implement this: return number of labels per sequence or bin"""
        pass
