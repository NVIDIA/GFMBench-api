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
Benchmark tasks package.

This package provides:
- base: Base classes for tasks and models
- concrete: Concrete task implementations (import per task module)

Example::

    from gfmbench_api.tasks import BaseGFMTask
    from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
"""

from gfmbench_api.tasks.base import (
    BaseGFMModel,
    BaseGFMTask,
    BaseGFMSupervisedClassificationTask,
    BaseGFMSupervisedRegressionTask,
    BaseGFMSupervisedSingleSeqTask,
    BaseGFMSupervisedTask,
    BaseGFMSupervisedVariantEffectTask,
    BaseGFMZeroShotGeneralIndelTask,
    BaseGFMZeroShotSNVTask,
    BaseGFMZeroShotTask,
    OutputSpatiality,
)

__all__ = [
    "BaseGFMModel",
    "BaseGFMTask",
    "BaseGFMSupervisedTask",
    "BaseGFMSupervisedClassificationTask",
    "BaseGFMSupervisedSingleSeqTask",
    "BaseGFMSupervisedVariantEffectTask",
    "BaseGFMSupervisedRegressionTask",
    "OutputSpatiality",
    "BaseGFMZeroShotTask",
    "BaseGFMZeroShotSNVTask",
    "BaseGFMZeroShotGeneralIndelTask",
]
