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
"""
GFMBench-API — genomic foundation model benchmarking middleware.

Import base task/model classes from this package; import concrete tasks
directly from their modules under ``gfmbench_api.tasks.concrete``.
"""

from gfmbench_api.tasks.base import (
    BaseGFMModel,
    BaseGFMTask,
    BaseGFMSupervisedClassificationTask,
    BaseGFMSupervisedRegressionTask,
    BaseGFMZeroShotGeneralIndelTask,
    BaseGFMZeroShotSNVTask,
    BaseGFMZeroShotTask,
    ClassificationMode,
    InputStructure,
    OutputSpatiality,
)

__all__ = [
    "BaseGFMModel",
    "BaseGFMTask",
    "BaseGFMSupervisedClassificationTask",
    "ClassificationMode",
    "InputStructure",
    "BaseGFMSupervisedRegressionTask",
    "OutputSpatiality",
    "BaseGFMZeroShotTask",
    "BaseGFMZeroShotSNVTask",
    "BaseGFMZeroShotGeneralIndelTask",
]
