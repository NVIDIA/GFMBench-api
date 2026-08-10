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
Concrete task implementations for GFM benchmarking.

Import tasks directly from their modules, e.g.::

    from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
    from gfmbench_api.tasks.concrete.songlab_clinvar_task import SonglabClinvarTask

Available task modules:

GUE (single-sequence classification):
- gue_promoter_all_task: GuePromoterAllTask
- gue_splice_site_task: GueSpliceSiteTask
- gue_tf_all_task: GueTranscriptionFactorTask

BEND (zero-shot SNV):
- bend_vep_expression_task: BendVEPExpression
- bend_vep_disease_task: BendVEPDisease

LRB:
- lrb_pathogenic_omim_task: LrbVariantEffectPathogenicOmimTask
- lrb_causal_eqtl_task: LRBCausalEqtlTask
- lrb_cage_task: LRBCagePredictionTask

TraitGym (zero-shot SNV):
- traitgym_complex_task: TraitGymComplexTask
- traitgym_mendelian_task: TraitGymMendelianTask

VariantBenchmarks (supervised variant effect):
- variant_benchmarks_coding_task: VariantBenchmarksCodingTask
- variant_benchmarks_non_coding_task: VariantBenchmarksNonCodingTask
- variant_benchmarks_expression_task: VariantBenchmarksExpressionTask
- variant_benchmarks_common_vs_rare_task: VariantBenchmarksCommonVsRareTask
- variant_benchmarks_meqtl_task: VariantBenchmarksMEQTLTask
- variant_benchmarks_sqtl_task: VariantBenchmarksSQTLTask

ClinVar / other zero-shot:
- clinvar_vepeval_task: VepevalClinvarTask
- clinvar_indel_task: IndelClinvarTask
- songlab_clinvar_task: SonglabClinvarTask
- brca1_task: BRCA1Task
- loleve_causal_eqtl_task: LoleveCausalEqtlTask
"""
