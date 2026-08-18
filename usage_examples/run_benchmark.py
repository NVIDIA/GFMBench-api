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
import argparse
import importlib
import logging
import os
import random
import sys
import time
from pathlib import Path

# Set before torch import for deterministic cuBLAS matmuls
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Safe with DataLoader num_workers > 0 after HuggingFace tokenizers are used in the main process
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure we import from the gfmbench_api_rep directory
# Add the gfmbench_api_rep root to the path if not already there
# run_benchmark.py is at: usage_examples/run_benchmark.py
# So we need to go up 2 levels: usage_examples -> gfmbench_api_rep
_standalone_root = Path(__file__).parent.parent
if str(_standalone_root) not in sys.path:
    sys.path.insert(0, str(_standalone_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gfmbench_api.utils import logutils

from gfmbench_api.benchmark_report import BenchmarkReport
from gfmbench_api.tasks.concrete.bend_vep_disease_task import BendVEPDisease
from gfmbench_api.tasks.concrete.bend_vep_expression_task import BendVEPExpression
from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from gfmbench_api.tasks.concrete.gue_splice_site_task import GueSpliceSiteTask
from gfmbench_api.tasks.concrete.gue_tf_all_task import GueTranscriptionFactorTask
from gfmbench_api.tasks.concrete.lrb_causal_eqtl_task import LRBCausalEqtlTask
from gfmbench_api.tasks.concrete.lrb_enhancer_task import LRBEnhancerTask
from gfmbench_api.tasks.concrete.lrb_histone_marks_task import LRBHistoneMarksTask
from gfmbench_api.tasks.concrete.lrb_pathogenic_omim_task import LrbVariantEffectPathogenicOmimTask
from gfmbench_api.tasks.concrete.songlab_clinvar_task import SonglabClinvarTask
from gfmbench_api.tasks.concrete.traitgym_complex_task import TraitGymComplexTask
from gfmbench_api.tasks.concrete.traitgym_mendelian_task import TraitGymMendelianTask
from gfmbench_api.tasks.concrete.variant_benchmarks_coding_task import VariantBenchmarksCodingTask
from gfmbench_api.tasks.concrete.variant_benchmarks_common_vs_rare_task import VariantBenchmarksCommonVsRareTask
from gfmbench_api.tasks.concrete.variant_benchmarks_expression_task import VariantBenchmarksExpressionTask
from gfmbench_api.tasks.concrete.variant_benchmarks_meqtl_task import VariantBenchmarksMEQTLTask
from gfmbench_api.tasks.concrete.variant_benchmarks_non_coding_task import VariantBenchmarksNonCodingTask
from gfmbench_api.tasks.concrete.variant_benchmarks_sqtl_task import VariantBenchmarksSQTLTask
from usage_examples.trainers import GFMFinetuner
from gfmbench_api.tasks.concrete.brca1_task import BRCA1Task
from gfmbench_api.tasks.concrete.clinvar_vepeval_task import VepevalClinvarTask
from gfmbench_api.tasks.concrete.clinvar_indel_task import IndelClinvarTask
from gfmbench_api.tasks.concrete.loleve_causal_eqtl_task import LoleveCausalEqtlTask

TASK_REGISTRY: dict[str, type] = {
    "vepeval_clinvar":                    VepevalClinvarTask,
    "brca1":                              BRCA1Task,
    "clinvar_indel":                      IndelClinvarTask,
    "loleve_causal_eqtl":                LoleveCausalEqtlTask,
    "lrb_variant_effect_causal_eqtl":    LRBCausalEqtlTask,
    "lrb_variant_effect_pathogenic_omim": LrbVariantEffectPathogenicOmimTask,
    "lrb_regulatory_element_enhancer":   LRBEnhancerTask,
    "lrb_chromatin_features_histone_marks": LRBHistoneMarksTask,
    "bend_variant_effects_disease":       BendVEPDisease,
    "bend_variant_effects_expression":    BendVEPExpression,
    "gue_promoter_all":                   GuePromoterAllTask,
    "gue_splice_site":                    GueSpliceSiteTask,
    "gue_transcription_factor":           GueTranscriptionFactorTask,
    "songlab_clinvar":                    SonglabClinvarTask,
    "traitgym_complex":                   TraitGymComplexTask,
    "traitgym_mendelian":                 TraitGymMendelianTask,
    "var_bench_coding_pathogenicity":     VariantBenchmarksCodingTask,
    "var_bench_non_coding_pathogenicity": VariantBenchmarksNonCodingTask,
    "var_bench_expression":               VariantBenchmarksExpressionTask,
    "var_bench_common_vs_rare":           VariantBenchmarksCommonVsRareTask,
    "var_bench_meqtl":                    VariantBenchmarksMEQTLTask,
    "var_bench_sqtl":                     VariantBenchmarksSQTLTask,
}

# Model metadata only — adapter modules are imported lazily via get_model_class().
MODEL_REGISTRY = {
    "DNABERT2": {
        "module": "usage_examples.sanity_models.dna_bert2_model",
        "class": "DNABERT2Model",
        "max_length": 2500,
    },
    "DNABERT": {
        "module": "usage_examples.sanity_models.dna_bert_model",
        "class": "DNABERTModel",
        "max_length": 500,
    },
    "Evo2": {
        "module": "usage_examples.sanity_models.evo2_model",
        "class": "Evo2BioNeMoModel",
        "max_length": 8192,
    },
    "NTv3_8M": {
        "module": "usage_examples.sanity_models.ntv3_model",
        "class": "NucleotideTransformerV3Model",
        "max_length": 8192,
        "model_kwargs": {"model_name": "NTv3_8M_pre", "use_autocast": False},
    },
    "NTv3_100M": {
        "module": "usage_examples.sanity_models.ntv3_model",
        "class": "NucleotideTransformerV3Model",
        "max_length": 8192,
        "model_kwargs": {"model_name": "NTv3_100M_pre", "use_autocast": True},
    },
    "Caduceus_Ps": {
        # Lives in Sense.ai (jepa_dna); requires PYTHONPATH to include Sense.ai/algo.
        "module": "jepa_dna.models.baseline_models.caduceus_model",
        "class": "CaduceusModel",
        "max_length": 8192,
        "model_kwargs": {
            "model_name": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        },
    },
    "HyenaDNA": {
        # LongSafari HyenaDNA tiny 16k (d_model=128); local GFMBench adapter.
        # Cap context at 8192 to match other long-seq baselines (Evo2/NTv3/Caduceus).
        "module": "usage_examples.sanity_models.hyena_dna_model",
        "class": "HyenaDNAModel",
        "max_length": 8192,
        "model_kwargs": {
            "model_name": "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
        },
    },
    "IsolatedMock": {
        "module": "usage_examples.sanity_models.isolated_mock_model",
        "class": "IsolatedMockModel",
        "max_length": 128,
    },
}


def get_model_class(model_name: str):
    """Import and return the model adapter class for the given registry key."""
    spec = MODEL_REGISTRY[model_name]
    try:
        module = importlib.import_module(spec["module"])
        return getattr(module, spec["class"])
    except ImportError as exc:
        raise ImportError(f"Failed to import {model_name} from {spec['module']}") from exc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run benchmark tasks on a model")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV path for the benchmark report (loads existing if present, saves here)"
    )
    parser.add_argument(
        "--report_algo_name",
        type=str,
        default="my_algo",
        help="Name for this model in the report"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to load (set to None to use default HuggingFace weights)"
    )
    parser.add_argument(
        "--linear_prob",
        action="store_true",
        help="If set, train only the projection layer for supervised tasks (linear probing). "
             "Otherwise, run full fine-tuning."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DNABERT2",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for benchmarking",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to fine-tune for supervised tasks (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic benchmark evaluation (default: 0)"
    )
    parser.add_argument(
        "--disable_safe_model_call",
        action="store_true",
        help="If set, model methods are called directly without try-except wrapper, "
             "allowing exceptions to propagate for debugging."
    )
    
    parser.add_argument(
        "--cache_size",
        type=float,
        default=None,
        help="Cache RAM limit in GB. 0 disables caching. None for unlimited.",
    )
    parser.add_argument(
        '--root_data_dir_path',
        type=str,
        required=True,
        help="Root data directory path. Datasets will be downloaded to this directory"
    )

    parser.add_argument(
        "--sanity_check_mode",
        action="store_true",
        help="If set, limit each dataset to 100 samples for quick testing.",
    )
    parser.add_argument(
        "--exclude_tasks",
        nargs="*",
        default=[],
        metavar="TASK",
        help="Task names to skip (e.g. --exclude_tasks vepeval_clinvar).",
    )
    return parser.parse_args(argv)


def set_seed(seed: int):
    """Set random seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except AttributeError:
        try:
            torch.set_deterministic(True)
        except AttributeError:
            pass

    os.environ["PYTHONHASHSEED"] = str(seed)


def _format_elapsed(seconds: float) -> str:
    """Human-readable duration for console logs."""
    if seconds >= 3600:
        hours, rem = divmod(int(seconds), 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}h {minutes}m {secs}s"
    if seconds >= 60:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def main(argv=None):
    # Initialize logging
    logutils.init_logger()
    
    args = parse_args(argv)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    seed = args.seed
    set_seed(seed)
    logging.info(f"Random seed set to: {seed} (deterministic evaluation)")
    
    # Path to the root data directory
    root_data_dir_path = args.root_data_dir_path
    
    # CSV path for the benchmark report (loads existing if present, saves here)
    csv_path = args.csv_path

    # Initialize the benchmark report
    report = BenchmarkReport(csv_path=csv_path)

    # Model configuration
    report_algo_name = args.report_algo_name
    
    checkpoint_path = args.checkpoint_path


    if args.model == "Evo2" and not args.linear_prob:
        raise ValueError("Evo2 does not support full fine-tuning. Use --linear_prob for Evo2 benchmarks.")
    ModelClass = get_model_class(args.model)
    max_length = MODEL_REGISTRY[args.model]["max_length"]
    model_init_kwargs = MODEL_REGISTRY[args.model].get("model_kwargs", {})
    
    logging.info(f"Model: {args.model}, Max sequence length: {max_length}")
    
    # Initialize model (will be reinitialized per task)
    model = ModelClass(device=device, max_length=max_length, **model_init_kwargs)
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)


    # Task configuration for DNABERT-2 (max 512 tokens by default, can extrapolate longer)
    # Supported keys: max_sequence_length, batch_size, num_workers, max_num_samples, disable_safe_model_call, cache_size
    # Set to None for models with no sequence length limit (e.g., HyenaDNA)
    task_config = {
        "max_sequence_length": max_length,
        "batch_size": 32,
        "num_workers": 0,
        "max_num_samples": None,
        "disable_safe_model_call": args.disable_safe_model_call,
        "cache_size": args.cache_size,
    }

    if args.cache_size is None:
        logging.warning(
            "cache_size not set; caches are unlimited (RAM). "
            "Pass --cache_size N (GB) to cap cache RAM."
        )
    else:
        logging.info(f"cache_size={args.cache_size} GB")

    if args.sanity_check_mode:
        task_config["max_num_samples"] = 100
        logging.info("SANITY CHECK MODE: Limiting to 100 samples per dataset")
    else:
        logging.info("Using full datasets (max_num_samples=None)")

    # Instantiate all tasks from the registry
    tasks = [
        cls(root_data_dir_path=root_data_dir_path, task_config=task_config)
        for name, cls in TASK_REGISTRY.items()
        if name not in args.exclude_tasks
    ]

    # Training parameters for fine-tuning tasks
    training_params = {
        "num_epochs": args.epochs,
        "lr": 3e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "only_proj_layer": args.linear_prob,
        "batch_size": 32,
        "num_workers": 0,
    }

    logging.info(f"********* Number of tasks: {len(tasks)} *********")
    logging.info(f"Model: {args.model}")
    logging.info(
        f"Training mode: {'Linear Probing' if training_params['only_proj_layer'] else 'Full Fine-tuning'}"
    )
    logging.info(f"Fine-tuning epochs: {args.epochs}")

    benchmark_start = time.perf_counter()

    # Run each task
    for task in tasks:
        task_start = time.perf_counter()
        task_name = task.get_task_name()
        logging.info(f"{'='*50}")
        logging.info(f"Running task: {task_name}")
        logging.info(f"{'='*50}")

        # Get task attributes
        task_attrs = task.get_task_attributes()
        has_finetuning_data = task_attrs.get("has_finetuning_data", False)

        # Re-seed before each task for deterministic model/projection initialization
        set_seed(seed)

        # Reinitialize model for each task to start fresh
        model = ModelClass(device=device, max_length=max_length, **model_init_kwargs)
        if checkpoint_path:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)

        if has_finetuning_data:
            hidden_dim = model.get_hidden_dim()
            task_type = task_attrs["task_type"]
            if task_type != "classification":
                raise NotImplementedError(
                    "The usage-example fine-tuner currently supports classification only."
                )
            classification_mode = task_attrs["classification_mode"]
            is_variant_effect = task_attrs["is_variant_effect_prediction"]
            num_labels = task_attrs["num_labels"]
            num_classes = task_attrs["num_classes"]
            num_outputs = (
                num_classes
                if num_labels == 1
                else num_labels if num_classes == 2 else num_labels * num_classes
            )
            
            train_dataset = task.get_finetune_dataset()

            generator = torch.Generator()
            generator.manual_seed(seed)

            train_loader = DataLoader(
                train_dataset,
                batch_size=training_params["batch_size"],
                shuffle=True,
                num_workers=training_params["num_workers"],
                generator=generator,
            )
            
            finetuner = GFMFinetuner(
                model=model,
                train_loader=train_loader,
                hidden_dim=hidden_dim,
                num_outputs=num_outputs,
                num_epochs=training_params["num_epochs"],
                lr=training_params["lr"],
                optimizer_name=training_params["optimizer"],
                weight_decay=training_params["weight_decay"],
                only_proj_layer=training_params["only_proj_layer"],
                classification_mode=classification_mode,
                num_labels=num_labels,
                num_classes=num_classes,
                is_variant_effect_prediction=is_variant_effect,
                cache_size=args.cache_size,
                device=device
            )
            
            logging.info("Fine-tuning...")
            test_model = finetuner.fine_tune()
        else:
            # For zero-shot tasks, use model directly
            logging.info("Zero-shot task detected. Skipping fine-tuning.")
            test_model = model

        # Evaluate on test set
        logging.info("Evaluating...")
        test_model.eval()
        scores = task.eval_test_set(test_model)
        logging.info(f"Test scores for {task_name}: {scores}")

        if hasattr(test_model, "clear_ref_cache"):
            test_model.clear_ref_cache()

        # Add scores to report and save immediately
        report.add_scores(task_name, report_algo_name, scores)
        report.save_csv()
        logging.info(f"Saved progress to: {csv_path}")

        task_elapsed = time.perf_counter() - task_start
        logging.info(
            f"Task '{task_name}' elapsed: {_format_elapsed(task_elapsed)} "
            f"({task_elapsed:.1f}s)"
        )

    total_elapsed = time.perf_counter() - benchmark_start
    logging.info(
        f"Total benchmark elapsed: {_format_elapsed(total_elapsed)} "
        f"({total_elapsed:.1f}s)"
    )

    # Print final report
    logging.info(f"{'='*50}")
    logging.info(f"Benchmark complete! Report saved to: {csv_path}")
    logging.info(f"{'='*50}")
    logging.info(report)


if __name__ == "__main__":
    main()
    logging.info("Benchmark completed!")

