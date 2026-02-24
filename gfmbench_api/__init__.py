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
