# GFMBench-API

GFMBench-API is an extensible benchmarking suite for assessing genomic foundation models (GFMs) across a diverse set of downstream tasks, including classification, variant effect prediction, and zero-shot evaluation.

## Quick Start

### Installation

1. Create a virtual environment (choose one option):

   **Option A: Using pip (venv)**
   ```bash
   python -m venv gfmbench_env
   source gfmbench_env/bin/activate  # On Windows: gfmbench_env\Scripts\activate
   ```

   **Option B: Using conda**
   ```bash
   conda create -n gfmbench_env python=3.10
   conda activate gfmbench_env
   ```

2. Install required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, GPU users) Check CUDA availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. (Optional, for DNABERT2 with flash attention):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

   Note: The requirements in `requirements.txt` are configured for running examples with DNABERT2 and DNABERT models.
   For other models or algorithms, additional dependencies may be required and should be installed separately.
---

## What’s Included?

GFMBench-API provides:

- A **core API package** for unified GFM evaluation (`gfmbench_api/`).
- A **suite of standard benchmark tasks** covering supervised and zero-shot scenarios.
- Consistent interfaces for models and tasks, enabling both out-of-the-box use and customized evaluation pipelines.
- Example scripts and templates (`usage_examples/`) to get started quickly or for rapid prototyping.

---

## Repository Organization

```
gfmbench_api_rep/
├── gfmbench_api/              # Main API package
│   ├── benchmark_report/      # CSV report utilities
│   ├── metrics/               # Built-in metrics (AUROC, AUPRC, etc.)
│   ├── tasks/                 # Task definitions
│   │   ├── base/              # Base task/model classes
│   │   └── concrete/          # 20+ ready-to-use tasks
│   └── utils/                 # Misc utilities (data I/O, download helpers)
├── usage_examples/            # Getting started scripts and toy models
│   ├── run_benchmark.py
│   ├── trainers/
│   └── sanity_models/
├── logs/                      # Logs (autocreated)
└── requirements.txt
```

---

## Supported Benchmarks & Tasks

GFMBench-API supports evaluation on **20 unique tasks**, grouped as:

### Supervised Classification & Variant Prediction

| Task Class                       | Description                             |
| --------------------------------- | --------------------------------------- |
| GuePromoterAllTask                | Binary classification of promoter vs non-promoter DNA sequences. |
| GueSpliceSiteTask                 | Three-class classification of splice sites as donor, acceptor, or non-splice. |
| GueTranscriptionFactorTask        | Binary classification of transcription factor binding sites from ChIP-seq data. |
| VariantBenchmarksCodingTask          | Binary classification of coding variants as benign or pathogenic. |
| VariantBenchmarksNonCodingTask       | Binary classification of non-coding variants as benign or pathogenic. |
| VariantBenchmarksExpressionTask      | Binary classification of variants affecting gene expression. |
| VariantBenchmarksCommonVsRareTask    | Binary classification distinguishing common variants from synthetic rare controls. |
| VariantBenchmarksMEQTLTask           | Binary classification of variants affecting DNA methylation rates. |
| VariantBenchmarksSQTLTask            | Binary classification of variants affecting alternative splicing. |
| LRBCausalEqtlTask                    | Binary classification of variants causally influencing gene expression with tissue context. |

### Zero-Shot Variant Effect Prediction

| Task Class                          | Description                                |
| -------------------------------------|--------------------------------------------|
| VepevalClinvarTask                   | Zero-shot pathogenicity prediction for ClinVar SNVs using embedding-distance scoring. |
| IndelClinvarTask                     | Zero-shot pathogenicity prediction for ClinVar insertions and deletions. |
| BendVEPExpression                    | Zero-shot prediction of expression effects for non-coding variants. |
| BendVEPDisease                       | Zero-shot prediction of disease effects for non-coding variants. |
| SonglabClinvarTask                | Zero-shot pathogenicity prediction for ClinVar SNVs using likelihood-based scoring. |
| BRCA1Task                         | Zero-shot prediction of functional impact for BRCA1 variants (LOF, intermediate, functional). |
| TraitGymComplexTask               | Zero-shot prediction of complex trait-associated variants. |
| TraitGymMendelianTask             | Zero-shot prediction of Mendelian disease-associated variants. |
| LrbVariantEffectPathogenicOmimTask   | Zero-shot prediction of pathogenic variants associated with Mendelian diseases. |
| LoleveCausalEqtlTask                 | Zero-shot prediction of causal expression-modulating variants (indels) in promoters. |

---

## How Model & Task Interfaces Work

### Task API

All task classes expose a consistent interface (see `gfmbench_api/tasks/base/`):

- `get_task_name()`
- `get_task_attributes()` &mdash; metadata (e.g. number of labels, dataset splits)
- `get_finetune_dataset()`
- `eval_test_set(model)`
- `eval_validation_set(model)`
- `eval_cross_validation_fold(model, train_indices)`

### Model Integration

Simply implement the methods below in your model class; inheritance is **not** required. The API uses duck typing and will only call methods needed by the specific tasks you run.

| Method Signature                                                                     | Description                            |
|:-------------------------------------------------------------------------------------|:------------------------------------------|
| `infer_sequence_to_labels_probs(sequences, ...)`                                     | Takes a list of DNA sequences and returns probabilities for each class label for classification tasks. |
| `infer_variant_ref_sequences_to_labels_probs(variant_sequences, ref_sequences, ...)`  | Takes lists of variant and reference sequences and returns probabilities for variant effect classification tasks. |
| `infer_sequence_to_sequence(sequences, ...)`                                         | Takes a list of DNA sequences and returns per-nucleotide probabilities, per-position embeddings, and a single sequence-level embedding. |
| `sequence_pos_to_prob_pos(sequences, pos)`                                           | Takes a list of DNA sequences and a position index, returns the corresponding output position indices accounting for tokenization differences. |
| `infer_masked_sequence_to_token_probs(sequences, variant_pos, variant_letters, reference_letters, ...)` | Takes sequences and masks the variant position, returns probabilities for the variant and reference nucleotides at the masked position. |

- Any not-implemented methods can simply return `None` and metrics depending on them will be skipped.
- See `gfmbench_api/tasks/base/base_gfm_model.py` for detailed docstrings.

---

## Example: Running Benchmarks

### The `usage_examples/run_benchmark.py` script

This reference script demonstrates a standard workflow:
- Loading a model (built-in or your own)
- Running a configurable set of tasks
- Supervised fine-tuning or zero-shot evaluation
- Auto-generating benchmark CSVs

#### Common CLI Arguments

| Argument             | Type    | Default | Notes                                             |
|----------------------|---------|---------|---------------------------------------------------|
| `--model`            | str     | DNABERT2| Model string key (see `MODEL_REGISTRY`)           |
| `--checkpoint_path`  | str     | None    | Path to model checkpoint                          |
| `--mlm_head_path`    | str     | None    | Optional: path to MLM head checkpoint             |
| `--report_algo_name` | str     | temp    | Column name for this model in benchmark CSV       |
| `--csv_path`         | str     | ...     | Where benchmark report is saved                   |
| `--linear_prob`      | flag    | False   | If set: train only projection layer               |
| `--epochs`           | int     | 3       | Num epochs for fine-tuning                        |
| `--disable_safe_model_call` | flag | False | Bypass try/except if set                          |

#### Key Variables to Adjust in `run_benchmark.py`

- **Root data directory**  
  ⚠️ Update around line 164:  
  ```python
  root_data_dir_path = "/path/to/your/data/"
  ```
- **Sanity check mode** (100-sample subsets)  
  ```python
  sanity_check_mode = True  # Set to False for full eval
  ```
- **Task configuration**  
  Edit parameters like `max_sequence_length`, `batch_size`.
- **Training parameters**  
  Change `num_epochs`, `learning_rate`, etc. in the training_params dict.
- **Task list**  
  Edit/add the tasks in the `tasks = [...]` list.
- **Model registry**  
  Add your models to the `MODEL_REGISTRY` dict.

#### Example usage

Default (no checkpoint):  
```bash
python usage_examples/run_benchmark.py \
    --model DNABERT2 \
    --report_algo_name dna_bert2_baseline \
    --csv_path results/baseline_results.csv
```

With a custom model checkpoint:  
```bash
python usage_examples/run_benchmark.py \
    --model DNABERT2 \
    --checkpoint_path /path/to/model.pt \
    --report_algo_name my_custom_model \
    --csv_path results/my_results.csv \
    --epochs 5
```

Linear probe (freeze backbone, train only projection):  
```bash
python usage_examples/run_benchmark.py \
    --model DNABERT2 \
    --linear_prob \
    --report_algo_name linear_probe \
    --csv_path results/linear_probe_results.csv
```

---

## Data Preparation

### Root Data Directory

All task data is stored under a single root directory (customize the path!):

```python
root_data_dir_path = "/path/to/your/data/"
```

### Task Data Structure

Each task creates a subdirectory, e.g.:

```
<root_data_dir_path>/
├── gue_promoter_all/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── clinvar_vepeval/
│   └── ...
└── ...
```

- Task subdirectory names correspond to class names.
- If files are missing, they will be automatically downloaded on first use.

#### Dataset Format

| Task Type        | Key columns                          | Example          |
|------------------|--------------------------------------|------------------|
| Classification   | `sequence`, `label`                  | `ATCCGA...`, 1   |
| Variant effect   | `ref_sequence`, `alt_sequence`, label| `A...T...`, 0    |

---

## Benchmark Output

Results are auto-saved in a single CSV with this schema:

| task             | metric       | model1 | model2 | ... |
|------------------|-------------|--------|--------|-----|
| gue_promoter_all | accuracy    | 0.85   | 0.82   | ... |
| gue_promoter_all | auroc       | 0.91   | 0.88   | ... |
| ...              | ...         | ...    | ...    | ... |

- Each (task, metric) is a row.
- Each model name (as passed to `--report_algo_name`) is a column.
- If a result is missing, its cell is `NO_RESULTS`.
- Incremental saving ensures safe restarts/interruption recovery.

Logs are written to `logs/` for all major steps (run progress, model errors, fine-tune loss curves, auto-downloading messages, etc).

---

## Customizing

### Example: Minimal Custom Pipeline

```python
from gfmbench_api.tasks.concrete.gue_promoter_all_task import GuePromoterAllTask
from gfmbench_api.benchmark_report import BenchmarkReport

task = GuePromoterAllTask(
    root_data_dir_path="/your/data/path",
    task_config={"max_sequence_length": 2500, "batch_size": 16},
)

my_model = MyCustomModel(device="cuda")  # Implements the required inference methods

# Optionally, supervised fine-tuning:
if task.get_task_attributes()["has_finetuning_data"]:
    dataset = task.get_finetune_dataset()
    # ... Your custom training code ...
    # After fine-tuning, wrap model with projection layer for classification (my_model = wrapped_model)

# Evaluate
my_model.eval()
results = task.eval_test_set(my_model)
print(results)

# Save results
report = BenchmarkReport(csv_path="results.csv")
report.add_scores("gue_promoter_all", "my_model", results)
report.save_csv()
```

---

## Adding Tasks

To add a new concrete task, inherit from the appropriate base class based on your task type and implement the required methods.

**Note:** If your task has unique functionality that doesn't fit the standard task types, you can inherit directly from `BaseGFMTask` and implement all required methods from scratch.

### Task Type Hierarchy

Choose the appropriate base class based on your task:

- **Supervised Single-Sequence Classification**: Inherit from `BaseGFMSupervisedSingleSeqTask`
  - For tasks with single DNA sequences and categorical labels (e.g., promoter prediction, splice site detection)
  
- **Supervised Variant Effect Prediction**: Inherit from `BaseGFMSupervisedVariantEffectTask`
  - For tasks with paired reference/variant sequences and categorical labels (e.g., pathogenic variant classification)
  
- **Zero-Shot SNV Variant Effect**: Inherit from `BaseGFMZeroShotSNVTask`
  - For zero-shot evaluation of single-nucleotide variants (SNVs) with equal-length reference and variant sequences
  
- **Zero-Shot General Indel**: Inherit from `BaseGFMZeroShotGeneralIndelTask`
  - For zero-shot evaluation of insertions/deletions with variable-length sequences

### Required Methods to Implement

All concrete tasks must implement:

1. **`get_task_name() -> str`**: Return the task name (must match the data directory name)
2. **`_get_default_max_seq_len() -> int`**: Return the default maximum sequence length
3. **`_create_datasets()` or `_create_test_dataset()`**: Create and return datasets
   - Supervised tasks: Return `Tuple[Optional[Dataset], Optional[Dataset], Dataset]` (train, validation, test)
   - Zero-shot tasks: Return `Dataset` (test only)
   - **Important:** When creating datasets, you must account for:
     - `self.max_sequence_length`: Truncate sequences if they exceed this length (use `truncate_sequence_from_ends()` from `gfmbench_api.utils.preprocutils`)
     - `self.max_num_samples`: Limit the number of samples per split if specified (for sanity testing with smaller datasets)
4. **`get_conditional_input_meta_data_frame() -> Optional[pd.DataFrame]`**: Return metadata DataFrame if task uses conditional inputs, otherwise return `None`

Additional methods for supervised tasks:

5. **`_get_num_labels() -> int`**: Return the number of classification labels

### Example: Supervised Single-Sequence Task

```python
from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import BaseGFMSupervisedSingleSeqTask
import os
import pandas as pd
import torch
import numpy as np

class MyCustomTask(BaseGFMSupervisedSingleSeqTask):
    def get_task_name(self) -> str:
        return "my_custom_task"
    
    def _get_default_max_seq_len(self) -> int:
        return 512
    
    def _get_num_labels(self) -> int:
        return 2  # Binary classification
    
    def _create_datasets(self):
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "dev.csv")) if os.path.exists(os.path.join(data_dir, "dev.csv")) else None
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        
        # Account for max_num_samples and max_sequence_length: limit samples and truncate sequences
        train_dataset = [(seq, label, np.array([])) for seq, label in zip(train_df['sequence'], train_df['label'])]
        validation_dataset = [(seq, label, np.array([])) for seq, label in zip(val_df['sequence'], val_df['label'])] if val_df is not None else None
        test_dataset = [(seq, label, np.array([])) for seq, label in zip(test_df['sequence'], test_df['label'])]
        
        return train_dataset, validation_dataset, test_dataset
    
    def get_conditional_input_meta_data_frame(self):
        return None
```

### Example: Zero-Shot SNV Task

```python
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
import os
import pandas as pd
import numpy as np

class MyZeroShotTask(BaseGFMZeroShotSNVTask):
    def get_task_name(self) -> str:
        return "my_zero_shot_task"
    
    def _get_default_max_seq_len(self) -> int:
        return 1024
    
    def _create_test_dataset(self):
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        
        # Account for max_num_samples and max_sequence_length: limit samples and truncate sequences
        test_dataset = [
            (var_seq, ref_seq, label, np.array([]))
            for var_seq, ref_seq, label in zip(test_df['variant_sequence'], test_df['reference_sequence'], test_df['label'])
        ]
        
        return test_dataset
    
    def _get_variant_position_in_sequence(self) -> int:
        return self.max_sequence_length // 2
    
    def get_conditional_input_meta_data_frame(self):
        return None
```

### Data Directory Structure

Place your task data in a subdirectory matching the task name:

```
<root_data_dir_path>/
└── my_custom_task/
    ├── train.csv
    ├── dev.csv (optional)
    └── test.csv
```

The task name returned by `get_task_name()` must exactly match this directory name.

**CSV Format Example:**

For supervised single-sequence tasks, CSV files should contain at least `sequence` and `label` columns:

```csv
sequence,label
ATCGATCGATCGATCG,1
GCTAGCTAGCTAGCTA,0
TTAACCGGAATTCCGG,1
```

For variant effect tasks, include `ref_sequence`, `alt_sequence`, and `label` columns:

```csv
ref_sequence,alt_sequence,label
ATCGATCGATCG,ATCGATCGATGG,1
GCTAGCTAGCTA,GCTAGCTAGCCA,0
```

---

## License

[Please add license information here if required.]

## Citation
@article{larey2026gfmbench,
  title={GFMBench-API: A Standardized Interface for Benchmarking Genomic Foundation Models},
  author={Larey, Ariel and Dahan, Elay and Amit Bleiweiss, Amit Bleiweiss and Kellerman, Raizy and Leib, Guy and Nayshool, Omri and Ofer, Dan and Zinger, Tal and Dominissini, Dan and Rechavi, Gideon and others},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
