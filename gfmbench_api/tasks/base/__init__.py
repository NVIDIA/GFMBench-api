# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
"""
Benchmark base classes for GFM evaluation.

This module exports:
- BaseGFMModel: Abstract base for genomic foundation models
- BaseGFMTask: Abstract base for all GFM tasks

Supervised task hierarchy:
- BaseGFMSupervisedMultiClassTask: Base for supervised multi-class tasks
  - BaseGFMSupervisedSingleSeqTask: For single sequence classification
  - BaseGFMSupervisedVariantEffectTask: For variant effect classification

Zero-shot task hierarchy:
- BaseGFMZeroShotTask: Base for zero-shot evaluation tasks
  - BaseGFMZeroShotSNVTask: For SNV variant effect tasks
  - BaseGFMZeroShotGeneralIndelTask: For general indel variant tasks
"""

from gfmbench_api.tasks.base.base_gfm_model import BaseGFMModel
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask

# Supervised task hierarchy
from gfmbench_api.tasks.base.base_gfm_supervised_multiclass_task import BaseGFMSupervisedMultiClassTask
from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import BaseGFMSupervisedSingleSeqTask
from gfmbench_api.tasks.base.base_gfm_supervised_variant_effect_task import BaseGFMSupervisedVariantEffectTask

# Zero-shot task hierarchy
from gfmbench_api.tasks.base.base_gfm_zero_shot_task import BaseGFMZeroShotTask
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.tasks.base.base_gfm_zeroshot_general_indel_task import BaseGFMZeroShotGeneralIndelTask

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
]
