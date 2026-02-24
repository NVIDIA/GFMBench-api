import os
import zipfile
from typing import Optional, Dict

import pandas as pd
import numpy as np
import torch

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils.fileutils import download_file_from_url, ensure_reference_genome
from gfmbench_api.utils.preprocutils import extract_snv_sequences_centered

class VepevalClinvarTask(BaseGFMZeroShotSNVTask):
    """
    ClinVar VEP-eval zero-shot SNV task.

    Uses the preprocessed benchmark parquet located at
    `data/clinvar_vep_eval/ClinVarBenchmark_subset_S1.parquet` and extracts
    sequence contexts from the hg38 reference genome.
    
    The parquet file is auto-downloaded from biorxiv supplementary materials if not present.
    The dataset is derived from SupplementaryTableS1.csv with rows containing any NaN values removed.

    Supports configuration via task_config dictionary:
    - max_sequence_length: Maximum sequence length (default: 1048576)
    - batch_size: Batch size for dataloaders (default: 32)
    - max_num_samples: Maximum number of samples to use (default: None = use all)
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the ClinVar VEP-eval task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """

        # Paths to reference genome and parquet file
        self.reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
        self.dataset_path = os.path.join(root_data_dir_path, "clinvar_vep_eval", "ClinVarBenchmark_subset_S1.parquet")

        # Initialize base class (creates datasets by calling _create_test_dataset)
        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        return "vepeval_clinvar"

    def _create_test_dataset(self):
        """
        Create test dataset from ClinVar VEP-eval benchmark.
        Extracts sequence contexts around each SNV from hg38 reference genome.
        Returns tuples of (variant_seq, reference_seq, label).
        """
        try:
            from pyfaidx import Fasta
        except ImportError:
            raise ImportError(
                "pyfaidx is required for extracting sequences from reference genome. "
                "Install with: pip install pyfaidx"
            )

        # Ensure reference genome exists (auto-download if missing)
        if not os.path.exists(self.reference_genome_path):
            print("Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)

        # Ensure dataset directory exists
        dataset_dir = os.path.dirname(self.dataset_path)
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and process parquet if not present
        if not os.path.exists(self.dataset_path):
            print("ClinVar VEP-eval parquet not found. Downloading from biorxiv...")
            self._download_and_prepare_dataset(dataset_dir)

        # Load parquet file
        print(f"Loading ClinVar VEP-eval dataset from: {self.dataset_path}")
        df = pd.read_parquet(self.dataset_path)
        print(f"Loaded {len(df)} rows from {self.dataset_path}")

        required_columns = ["chrom", "pos", "ref", "alt", "label"]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                "ClinVar VEP-eval dataset missing required columns. "
                f"Expected {required_columns}, but missing {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Apply max_num_samples if specified (early slice for efficiency)
        if self.max_num_samples is not None:
            df = df.iloc[: self.max_num_samples]
            print(f"[Fast run] Using first {len(df)} SNVs (max_num_samples={self.max_num_samples}).")

        # Load reference genome
        print(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)

        # Extract sequences using shared utility function
        variant_sequences, reference_sequences, labels, skipped = extract_snv_sequences_centered(
            df=df,
            genome=genome,
            max_sequence_length=self.max_sequence_length,
            chrom_col="chrom",
            pos_col="pos",
            ref_col="ref",
            alt_col="alt",
            label_col="label",
            use_logging=False,  # Use print for consistency with existing behavior
        )

        if skipped > 0:
            print(f"Skipped {skipped} variants due to extraction issues or non-SNV records.")

        if len(labels) == 0:
            raise ValueError("No valid samples after sequence extraction from ClinVar VEP-eval dataset.")

        self.labels = torch.tensor(labels, dtype=torch.long)
        # Create dataset: (variant_sequence, reference_sequence, label, conditional_input) tuples
        # conditional_input is empty numpy array since this task has no conditional metadata
        test_dataset = [
            (var_seq, ref_seq, label, np.array([], dtype=np.float32))
            for var_seq, ref_seq, label in zip(variant_sequences, reference_sequences, self.labels)
        ]

        print(f"ClinVar VEP-eval test dataset ready with {len(test_dataset)} samples.")
        return test_dataset

    def _download_and_prepare_dataset(self, dataset_dir: str) -> None:
        """
        Download ClinVar VEP-eval benchmark from biorxiv and prepare parquet.
        Extracts SupplementaryTableS1.csv from the zip, applies dropna, and saves as parquet.
        
        Note: If automated download fails due to server restrictions, you can:
        1. Download manually from: https://www.biorxiv.org/content/10.1101/2025.09.05.674459v1
        2. Look for "SupplementaryTableS1.csv" in the supplementary materials
        3. Place the zip file in: data/clinvar_vep_eval/media-1.zip
        4. Or place the CSV in: data/clinvar_vep_eval/SupplementaryTableS1.csv
        5. Or place the parquet directly at: data/clinvar_vep_eval/ClinVarBenchmark_subset_S1.parquet
        """
        zip_url = "https://www.biorxiv.org/content/biorxiv/early/2025/09/10/2025.09.05.674459/DC1/embed/media-1.zip?download=true"
        
        # Check if zip file already exists in dataset directory
        local_zip_path = os.path.join(dataset_dir, "media-1.zip")
        
        if os.path.exists(local_zip_path):
            print(f"Found existing zip file: {local_zip_path}")
            print("Using local zip file (skipping download)")
            zip_path_to_use = local_zip_path
        else:
            print("Downloading ClinVar VEP-eval dataset from biorxiv...")
            print("Note: If this fails, please download manually (see error message for details)")
            
            urls_to_try = [zip_url]
            
            download_success = False
            for attempt_url in urls_to_try:
                try:
                    # Download zip to dataset directory (not temporary)
                    download_file_from_url(attempt_url, local_zip_path)
                    print(f"Downloaded zip file to: {local_zip_path}")
                    zip_path_to_use = local_zip_path
                    download_success = True
                    break
                except Exception as e:
                    print(f"Failed with URL: {attempt_url}")
                    print(f"Error: {str(e)}")
                    continue
            
            if not download_success:
                # Provide helpful manual download instructions
                raise RuntimeError(
                    f"Failed to automatically download ClinVar VEP-eval dataset from biorxiv.\n"
                    f"\n"
                    f"To manually download the dataset:\n"
                    f"1. Visit: https://www.biorxiv.org/content/10.1101/2025.09.05.674459v1\n"
                    f"2. Download the supplementary materials zip file\n"
                    f"3. Place the zip at: {local_zip_path}\n"
                    f"   OR extract SupplementaryTableS1.csv and place at: {os.path.dirname(self.dataset_path)}/SupplementaryTableS1.csv\n"
                    f"   OR place the parquet directly at: {self.dataset_path}\n"
                    f"\n"
                    f"Then run the script again.\n"
                )
        
        # Extract and process the zip file
        try:
            # Extract CSV from zip
            print("Extracting SupplementaryTableS1.csv from zip...")
            with zipfile.ZipFile(zip_path_to_use, 'r') as zip_ref:
                # Find the CSV file in the zip (it might be nested)
                csv_files = [f for f in zip_ref.namelist() if f.endswith('SupplementaryTableS1.csv')]
                if not csv_files:
                    raise FileNotFoundError(
                        f"SupplementaryTableS1.csv not found in downloaded zip. "
                        f"Files in zip: {zip_ref.namelist()}"
                    )
                
                csv_filename = csv_files[0]
                print(f"Found CSV: {csv_filename}")
                
                # Read CSV directly from zip
                with zip_ref.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file)
            
            # Normalize column names to match extraction logic
            df = df.rename(
                columns={
                    "#CHROM": "chrom",
                    "chrom": "chrom",
                    "POS": "pos",
                    "REF": "ref",
                    "ALT": "alt",
                    "ClinVar_label": "label",
                }
            )

            # Clean data: drop rows with any NA values
            print(f"Original dataset: {len(df)} rows")

            # Standardize data types and format 
            df = df.copy()
            df["chrom"] = df["chrom"].astype(str).apply(lambda c: f"chr{c}" if not str(c).startswith("chr") else c)
            df["pos"] = df["pos"].astype(int)
            df["ref"] = df["ref"].astype(str).str.upper()
            df["alt"] = df["alt"].astype(str).str.upper()
            if df["label"].dtype == bool:
                df["label"] = df["label"].astype(int)

            # # Keep only SNVs (single nucleotide ref/alt)
            
            snv_mask = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
            skipped = int((~snv_mask).sum())
            if skipped > 0:
                print(f"Skipping {skipped} non-SNV variants.")
            df = df[snv_mask]
            ## paper's evaluation dropped rows with nans from any model. We do same (but seem to drop more, as paper had ~ 259K/261K retained roughly)
            ## https://github.com/Brandes-Lab/VEP-eval/blob/main/VEP_AUROC_figure.ipynb
            ## NOTE: WARNING: Some of these models are coding/protein only, we exclude them from the dropping.
            df = df.dropna(subset=['AlphaGenome_quantile','Evo2',
                        'DNABERT2', 'Nucleotide_Transformer', 'PhyloP', 'GPN_MSA', 'Rule_based',
                        #   'ESM1v',  'ESM1b', 'ESM2',     'PrimateAI_3D',        'AlphaMissense', c## protein only/coding models, exclude from the dropna criteria
                        'PhyloGPN'],axis=0, how="any")
            
            print(f"After removing rows with missing values: {len(df)} rows")
            
            # Save as parquet
            os.makedirs(dataset_dir, exist_ok=True)
            df.to_parquet(self.dataset_path, index=False)
            print(f"Dataset saved to: {self.dataset_path}")
            
            print(f"Zip file retained at: {local_zip_path}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to process zip file at {zip_path_to_use}\n"
                f"Error: {str(e)}\n"
                f"Please check that the zip file is valid and contains SupplementaryTableS1.csv"
            )

    def _get_variant_position_in_sequence(self) -> int:
        """Return the position of the SNV in the sequence."""
        return self.max_sequence_length // 2

    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length (1048576bp)."""
        return 1048576

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None