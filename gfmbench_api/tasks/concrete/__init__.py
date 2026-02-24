"""
Concrete task implementations for GFM benchmarking.

GUE (Genomic Understanding Evaluation) tasks - Single sequence classification:
- GuePromoterAllTask: Promoter prediction (2-class)
- GueSpliceSiteTask: Splice site prediction (3-class)
- GueTranscriptionFactorTask: Transcription factor binding (2-class)

BEND (Benchmark for Evaluating Nucleotide-level DNA) tasks - Zero-shot SNV:
- BendVEPExpression: Variant effect on expression
- BendVEPDisease: Variant effect on disease

LRB (Long Range Benchmark) tasks - Zero-shot SNV:
- LrbVariantEffectPathogenicOmimTask: OMIM pathogenic variant prediction

TraitGym tasks - Zero-shot SNV:
- TraitGymComplexTask: Complex trait variant prediction
- TraitGymMendelianTask: Mendelian disease variant prediction

VariantBenchmarks tasks - Supervised variant effect:
- VariantBenchmarksCodingTask: Coding variant effect
- VariantBenchmarksNonCodingTask: Non-coding variant effect
- VariantBenchmarksExpressionTask: Expression variant effect
- VariantBenchmarksCommonVsRareTask: Common vs rare variant classification
- VariantBenchmarksMEQTLTask: meQTL variant effect
- VariantBenchmarksSQTLTask: sQTL variant effect
"""

# GUE tasks (single sequence classification)
from .gue_promoter_all_task import GuePromoterAllTask
from .gue_splice_site_task import GueSpliceSiteTask
from .gue_tf_all_task import GueTranscriptionFactorTask

# BEND tasks (zero-shot SNV)
from .bend_vep_expression_task import BendVEPExpression
from .bend_vep_disease_task import BendVEPDisease

# LRB tasks (zero-shot SNV)
from .lrb_pathogenic_omim_task import LrbVariantEffectPathogenicOmimTask

# TraitGym tasks (zero-shot SNV)
from .traitgym_complex_task import TraitGymComplexTask
from .traitgym_mendelian_task import TraitGymMendelianTask

# VariantBenchmarks tasks (supervised variant effect)
from .variant_benchmarks_coding_task import VariantBenchmarksCodingTask
from .variant_benchmarks_non_coding_task import VariantBenchmarksNonCodingTask
from .variant_benchmarks_expression_task import VariantBenchmarksExpressionTask
from .variant_benchmarks_common_vs_rare_task import VariantBenchmarksCommonVsRareTask
from .variant_benchmarks_meqtl_task import VariantBenchmarksMEQTLTask
from .variant_benchmarks_sqtl_task import VariantBenchmarksSQTLTask

__all__ = [
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
