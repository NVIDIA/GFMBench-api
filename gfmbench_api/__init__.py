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
Benchmarks package for GFM evaluation.

This package provides:
- Base classes for tasks and models
- Metric implementations
- Concrete task implementations
"""

from gfmbench_api.tasks.base import (
    # Core classes
    BaseGFMModel,
    BaseGFMTask,
    # Supervised task hierarchy
    BaseGFMSupervisedMultiClassTask,
    BaseGFMSupervisedSingleSeqTask,
    BaseGFMSupervisedVariantEffectTask,
    # Zero-shot task hierarchy
    BaseGFMZeroShotTask,
    BaseGFMZeroShotSNVTask,
    BaseGFMZeroShotGeneralIndelTask,
)

from gfmbench_api.tasks.concrete import (
    # GUE tasks
    GuePromoterAllTask,
    GueSpliceSiteTask,
    GueTranscriptionFactorTask,
    # BEND tasks
    BendVEPExpression,
    BendVEPDisease,
    # LRB tasks
    LrbVariantEffectPathogenicOmimTask,
    # TraitGym tasks
    TraitGymComplexTask,
    TraitGymMendelianTask,
    # VariantBenchmarks tasks
    VariantBenchmarksCodingTask,
    VariantBenchmarksNonCodingTask,
    VariantBenchmarksExpressionTask,
    VariantBenchmarksCommonVsRareTask,
    VariantBenchmarksMEQTLTask,
    VariantBenchmarksSQTLTask,
)

__all__ = [
    # Core classes
    "BaseGFMModel",
    "BaseGFMTask",
    # Supervised task hierarchy
    "BaseGFMSupervisedMultiClassTask",
    "BaseGFMSupervisedSingleSeqTask",
    "BaseGFMSupervisedVariantEffectTask",
    # Zero-shot task hierarchy
    "BaseGFMZeroShotTask",
    "BaseGFMZeroShotSNVTask",
    "BaseGFMZeroShotGeneralIndelTask",
    # GUE tasks
    "GuePromoterAllTask",
    "GueSpliceSiteTask",
    "GueTranscriptionFactorTask",
    # BEND tasks
    "BendVEPExpression",
    "BendVEPDisease",
    # LRB tasks
    "LrbVariantEffectPathogenicOmimTask",
    # TraitGym tasks
    "TraitGymComplexTask",
    "TraitGymMendelianTask",
    # VariantBenchmarks tasks
    "VariantBenchmarksCodingTask",
    "VariantBenchmarksNonCodingTask",
    "VariantBenchmarksExpressionTask",
    "VariantBenchmarksCommonVsRareTask",
    "VariantBenchmarksMEQTLTask",
    "VariantBenchmarksSQTLTask",
]
