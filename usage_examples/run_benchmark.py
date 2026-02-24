import argparse
import logging
import sys
from pathlib import Path

# Ensure we import from the gfmbench_api_rep directory
# Add the gfmbench_api_rep root to the path if not already there
# run_benchmark.py is at: usage_examples/run_benchmark.py
# So we need to go up 2 levels: usage_examples -> gfmbench_api_rep
_standalone_root = Path(__file__).parent.parent
if str(_standalone_root) not in sys.path:
    sys.path.insert(0, str(_standalone_root))

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
from usage_examples.sanity_models.dna_bert2_model import DNABERT2Model
from usage_examples.sanity_models.dna_bert_model import DNABERTModel
from gfmbench_api.tasks.concrete.brca1_task import BRCA1Task
from gfmbench_api.tasks.concrete.clinvar_vepeval_task import VepevalClinvarTask
from gfmbench_api.tasks.concrete.clinvar_indel_task import IndelClinvarTask
from gfmbench_api.tasks.concrete.loleve_causal_eqtl_task import LoleveCausalEqtlTask

# Mapping of model names to model classes and their default max sequence lengths
MODEL_REGISTRY = {
    "DNABERT2": {"class": DNABERT2Model, "max_length": 2500},
    "DNABERT": {"class": DNABERTModel, "max_length": 500},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark tasks on a model")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/efs/alarey/projects/sense/resutls/jepa/benchmark_DNABERT2_160225.csv",
        help="CSV path for the benchmark report (loads existing if present, saves here)"
    )
    parser.add_argument(
        "--report_algo_name",
        type=str,
        default="temp",
        help="Name for this model in the report"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to load (set to None to use default HuggingFace weights)"
    )
    parser.add_argument(
        "--mlm_head_path",
        type=str,
        default=None,
        help="Path to checkpoint containing MLM head weights. If provided along with "
             "--checkpoint_path, first loads full checkpoint, then overwrites MLM head "
             "with weights from this path. Only mlm_head_state_dict is loaded; other weights are ignored."
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
        help="Model to use for benchmarking. Supported: DNABERT2, DNABERT"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to fine-tune for supervised tasks (default: 3)"
    )
    parser.add_argument(
        "--disable_safe_model_call",
        action="store_true",
        help="If set, model methods are called directly without try-except wrapper, "
             "allowing exceptions to propagate for debugging."
    )
    return parser.parse_args()


def load_mlm_head_only(model, mlm_head_path: str):
    """
    Load only the MLM head weights from a checkpoint.
    
    Handles checkpoints that contain:
    - mlm_head_state_dict: Load MLM head weights directly
    - model_state_dict + mlm_head_state_dict: Load only mlm_head_state_dict, ignore model weights
    - mlm_model_state_dict (legacy): Extract and load only cls.* weights
    
    Args:
        model: DNABERT2Model instance
        mlm_head_path: Path to checkpoint file containing MLM head weights
    """
    logging.info(f"Loading MLM head only from: {mlm_head_path}")
    state = torch.load(mlm_head_path, map_location=model.device)
    
    if 'mlm_head_state_dict' in state:
        # New format: mlm_head_state_dict is directly available
        model.mlm_head.load_state_dict(state['mlm_head_state_dict'])
        model.mlm_head_loaded = True
        logging.info("Loaded MLM head weights from mlm_head_state_dict")
        
    elif 'mlm_model_state_dict' in state:
        # Legacy format: extract cls.* weights from full AutoModelForMaskedLM state
        mlm_state = state['mlm_model_state_dict']
        mlm_head_state = {}
        for key, value in mlm_state.items():
            if key.startswith('cls.'):
                mlm_head_state[key[4:]] = value  # Remove 'cls.' prefix
        
        if mlm_head_state:
            model.mlm_head.load_state_dict(mlm_head_state, strict=False)
            model.mlm_head_loaded = True
            logging.info("Loaded MLM head weights from legacy mlm_model_state_dict")
        else:
            logging.warning("No MLM head weights found in legacy checkpoint")
    else:
        logging.error(f"No MLM head weights found in {mlm_head_path}")
        logging.error(f"  Available keys: {list(state.keys())}")
        raise ValueError("No MLM head weights found in MLM head checkpoint")


def main():
    # Initialize logging
    logutils.init_logger()
    
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # =========================================================================
    # Sanity check mode: use minimal data for quick testing
    # =========================================================================
    sanity_check_mode = False  # Set to True for quick testing with 100 samples
    
    # Path to the root data directory
    root_data_dir_path = "/efs/alarey/projects/sense/data/"
    
    # CSV path for the benchmark report (loads existing if present, saves here)
    csv_path = args.csv_path

    # Initialize the benchmark report
    report = BenchmarkReport(csv_path=csv_path)

    # Model configuration
    report_algo_name = args.report_algo_name
    
    # Checkpoint paths
    checkpoint_path = args.checkpoint_path
    mlm_head_path = args.mlm_head_path
    
    # Get model class and max_length from registry
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}. Supported models: {list(MODEL_REGISTRY.keys())}")
    ModelClass = MODEL_REGISTRY[args.model]["class"]
    max_length = MODEL_REGISTRY[args.model]["max_length"]
    
    logging.info(f"Model: {args.model}, Max sequence length: {max_length}")
    
    # Initialize model (will be reinitialized per task)
    model = ModelClass(device=device, max_length=max_length)
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    if mlm_head_path:
        load_mlm_head_only(model, mlm_head_path)
    
    # Task configuration for DNABERT-2 (max 512 tokens by default, can extrapolate longer)
    # Supported keys: max_sequence_length, batch_size, max_num_samples, disable_safe_model_call
    # Set to None for models with no sequence length limit (e.g., HyenaDNA)
    task_config = {
        "max_sequence_length": max_length,
        "batch_size": 16,
        "max_num_samples": None,
        "disable_safe_model_call": args.disable_safe_model_call,
    }
    
    # Apply sanity check mode
    if sanity_check_mode:
        task_config["max_num_samples"] = 100
        logging.info("SANITY CHECK MODE: Limiting to 100 samples per dataset")

    # Define all tasks to run
    tasks = [
        VepevalClinvarTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        IndelClinvarTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        LoleveCausalEqtlTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        BRCA1Task(root_data_dir_path=root_data_dir_path, task_config=task_config),
        GueTranscriptionFactorTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        GuePromoterAllTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        GueSpliceSiteTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        BendVEPExpression(root_data_dir_path=root_data_dir_path, task_config=task_config),
        BendVEPDisease(root_data_dir_path=root_data_dir_path, task_config=task_config),
        SonglabClinvarTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksCodingTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksNonCodingTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksExpressionTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksCommonVsRareTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksMEQTLTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        VariantBenchmarksSQTLTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        TraitGymComplexTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        TraitGymMendelianTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        LrbVariantEffectPathogenicOmimTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
        LRBCausalEqtlTask(root_data_dir_path=root_data_dir_path, task_config=task_config),
    ]

    # Training parameters for fine-tuning tasks
    training_params = {
        "num_epochs": args.epochs,
        "lr": 3e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "only_proj_layer": args.linear_prob,
        "batch_size": 8,
    }

    logging.info(f"********* Number of tasks: {len(tasks)} *********")
    logging.info(f"Model: {args.model}")
    logging.info(f"Training mode: {'Linear Probing' if args.linear_prob else 'Full Fine-tuning'}")
    logging.info(f"Fine-tuning epochs: {args.epochs}")

    # Run each task
    for task in tasks:
        task_name = task.get_task_name()
        logging.info(f"{'='*50}")
        logging.info(f"Running task: {task_name}")
        logging.info(f"{'='*50}")

        # Get task attributes
        task_attrs = task.get_task_attributes()
        has_finetuning_data = task_attrs.get("has_finetuning_data", False)

        # Reinitialize model for each task to start fresh
        model = ModelClass(device=device, max_length=max_length)
        if checkpoint_path:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
        if mlm_head_path:
            logging.info(f"Loading MLM head from: {mlm_head_path}")
            load_mlm_head_only(model, mlm_head_path)

        if has_finetuning_data:
            # Fine-tune for classification tasks
            hidden_dim = model.get_hidden_dim()
            num_labels = task_attrs["num_labels"]
            is_variant_effect = task_attrs.get("is_variant_effect_prediction", False)
            
            train_dataset = task.get_finetune_dataset()
            train_loader = DataLoader(
                train_dataset, 
                batch_size=training_params.get("batch_size", 32), 
                shuffle=True
            )
            
            finetuner = GFMFinetuner(
                model=model,
                train_loader=train_loader,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                num_epochs=training_params["num_epochs"],
                lr=training_params["lr"],
                optimizer_name=training_params["optimizer"],
                weight_decay=training_params["weight_decay"],
                only_proj_layer=training_params["only_proj_layer"],
                is_variant_effect_prediction=is_variant_effect,
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

        # Add scores to report and save immediately
        report.add_scores(task_name, report_algo_name, scores)
        report.save_csv()
        logging.info(f"Saved progress to: {csv_path}")

    # Print final report
    logging.info(f"{'='*50}")
    logging.info(f"Benchmark complete! Report saved to: {csv_path}")
    logging.info(f"{'='*50}")
    logging.info(report)


if __name__ == "__main__":
    main()
    logging.info("Benchmark completed!")

