# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/songlab/clinvar — MIT
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from pyfaidx import Fasta
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant


class SonglabClinvarTask(BaseGFMZeroShotSNVTask):
    """
    ClinVar zero-shot SNV task.

    Loads ClinVar test split from HuggingFace (songlab/clinvar) and extracts
    sequences around SNVs using the hg38 reference. The dataset is saved as
    a parquet cache at `data/clinvar/clinvar_data.parquet` after the first download.

    Supports configuration via task_config dictionary:
    - max_sequence_length: Maximum sequence length
    - batch_size: Batch size for dataloaders (default: 32)
    - max_num_samples: Maximum number of samples to use (default: None = use all)
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the ClinVar task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """
        # Allow user to pass '~' paths
        root_data_dir_path = os.path.expanduser(root_data_dir_path)

        # Set HuggingFace dataset name and reference genome path
        # These must be set before calling super().__init__() since _create_test_dataset needs them
        self.hf_dataset_name = "songlab/clinvar"
        self.reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")

        # Call parent initialization
        # This will parse task_config to set self.max_sequence_length, self.batch_size, self.max_num_samples
        # and then call self._create_datasets() -> self._create_test_dataset()
        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        """Return task name (identical to data directory name)."""
        return "songlab_clinvar"

    def _get_default_max_seq_len(self) -> int:
        """Return the task's default maximum sequence length.
        """
        return 1048576

    def _create_test_dataset(self) -> Dataset:
        """
        Create test dataset from HuggingFace ClinVar dataset.
        Extracts sequence contexts around each SNV from reference genome.
        Returns tuples of (variant_seq, reference_seq, label).
        """
        # Ensure reference genome exists (auto-download if missing)
        if not os.path.exists(self.reference_genome_path):
            logging.info("Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)

        # Ensure data dir exists
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        os.makedirs(data_dir, exist_ok=True)
        parquet_path = os.path.join(data_dir, "clinvar_data.parquet")

        # Load cached parquet if available, otherwise download from HF
        if os.path.exists(parquet_path):
            logging.info(f"Loading ClinVar dataset from cache: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        else:
            logging.info(f"Downloading ClinVar dataset from {self.hf_dataset_name}...")
            try:
                ds = load_dataset(self.hf_dataset_name, split="test")
                df = ds.to_pandas()
                logging.info(f"Saving ClinVar dataset to: {parquet_path}")
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download ClinVar dataset from {self.hf_dataset_name}.\n"
                    f"Error: {str(e)}\n"
                    f"Please ensure the datasets library is installed and you have internet access."
                )

        logging.info(f"Loaded {len(df)} samples from ClinVar dataset")
        # --- OPTIMIZATION: Slice early to save processing time if using subset ---
        if self.max_num_samples is not None:
            logging.info(f"[Fast run] Slicing dataframe to first {self.max_num_samples} SNVs before extraction.")
            df: pd.DataFrame = df.head(self.max_num_samples)

        # Required columns
        required_columns = ["chrom", "pos", "ref", "alt", "label"]
        if not all(col in df.columns for col in required_columns):
            available_cols = list(df.columns)
            raise ValueError(
                f"ClinVar dataset missing required columns. Expected {required_columns}, but found {available_cols}.\n"
                f"First few rows:\n{df.head()}"
            )

        # Load reference genome
        logging.info(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)

        # Calculate flank size from max_sequence_length
        # Use window_size // 2 to match GPN implementation
        flank_size = self.max_sequence_length // 2

        # Extract sequences from genome (group by chromosome for efficiency)
        logging.info(f"Extracting sequences (context window: {self.max_sequence_length}bp, flank: {flank_size}bp)...")
        reference_sequences = []
        variant_sequences = []
        labels = []
        skipped = 0

        # Normalize chromosome names and types, convert labels to ints
        df = df.copy()
        df['chrom'] = df['chrom'].astype(str).apply(lambda c: f"chr{c}" if not str(c).startswith('chr') else c)
        df['pos'] = df['pos'].astype(int)
        df['ref'] = df['ref'].astype(str).str.upper()
        df['alt'] = df['alt'].astype(str).str.upper()
        if df['label'].dtype == bool:
            df['label'] = df['label'].astype(int)

        # Process per chromosome to avoid repeated pyfaidx calls
        for chrom, group in df.groupby('chrom'):
            try:
                chrom_obj = genome[chrom]
            except KeyError:
                skipped += len(group)
                continue
            except Exception:
                skipped += len(group)
                continue

            # Process each variant with padding support
            for _, row in group.iterrows():
                pos = int(row['pos'])  # 1-based VCF position
                ref_allele = str(row['ref']).upper()
                alt_allele = str(row['alt']).upper()
                label = int(row['label']) if isinstance(row['label'], (bool, np.bool_)) else row['label']

                # Convert to 0-based
                pos_0 = pos - 1
                
                # Extract reference sequence using padding function (handles chromosome boundaries)
                try:
                    ref_seq = pad_sequence_centered_variant(
                        chromosome=chrom_obj,
                        variant_pos_0based=pos_0,
                        max_sequence_length=self.max_sequence_length,
                        variant_pos_in_seq=flank_size
                    )
                except Exception as e:
                    logging.warning(f"Error extracting sequence for {chrom}:{pos}. {str(e)}. Skipping.")
                    skipped += 1
                    continue

                # Verify sequence length
                if len(ref_seq) != self.max_sequence_length:
                    skipped += 1
                    continue

                # Verify variant is at expected center position
                variant_pos_in_window = flank_size
                
                # Check if padding character is at variant position (shouldn't happen, but safety check)
                if variant_pos_in_window < len(ref_seq) and ref_seq[variant_pos_in_window] == 'P':
                    logging.warning(f"Variant position at {chrom}:{pos} falls in padding region. Skipping.")
                    skipped += 1
                    continue

                # Validate reference allele matches genome
                if variant_pos_in_window < len(ref_seq):
                    if ref_seq[variant_pos_in_window] != ref_allele:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

                var_seq = ref_seq[:variant_pos_in_window] + alt_allele + ref_seq[variant_pos_in_window + 1:]

                reference_sequences.append(ref_seq)
                variant_sequences.append(var_seq)
                labels.append(label)

        if skipped > 0:
            logging.warning(f"Skipped {skipped} variants due to extraction issues or allele mismatch")

        logging.info(f"After extraction: {len(labels)} samples with valid sequences")

        if len(labels) == 0:
            raise ValueError(
                "No valid samples after extracting sequences from reference genome. "
                "Check that hg38.fa is compatible with the ClinVar dataset."
            )

        # Save labels and create dataset tuples
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Create dataset: (variant_sequence, reference_sequence, label, conditional_input) tuples
        test_dataset = [
            (var_seq, ref_seq, label, np.array([]))
            for var_seq, ref_seq, label in zip(variant_sequences, reference_sequences, self.labels)
        ]

        logging.info(f"ClinVar test dataset ready with {len(test_dataset)} valid samples (Skipped/Dropped {skipped} total).")
        return test_dataset

    def _get_variant_position_in_sequence(self) -> int:
        """Return the position of the SNV in the sequence.
        
        Uses window_size // 2 to match GPN paper implementation.
        """
        return self.max_sequence_length // 2

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None