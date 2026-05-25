# -*- coding: utf-8 -*-
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

# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://huggingface.co/datasets/songlab/TraitGym — MIT
"""TraitGym.ipynb

Original file is located at
    https://colab.research.google.com/github/songlab-cal/TraitGym/blob/main/TraitGym.ipynb

# TraitGym (https://colab.research.google.com/github/songlab-cal/TraitGym/blob/main/TraitGym.ipynb)
In this example we load the Mendelian traits (or complex traits) dataset and run variant effect prediction (VEP) based on euclidean distance of GPN-Animal-Promoter embeddings of the reference and alternate sequences.

## Setup
"""

# !pip install -q pyfaidx s3fs git+https://github.com/songlab-cal/gpn.git
# !pip install -q -U transformers datasets

# dataset_path = "songlab/TraitGym"
# # dataset_config = "mendelian_traits"
# dataset_config = "complex_traits"
import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pyfaidx import Fasta
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant


def _load_traitgym_dataset(
    root_data_dir_path: str,
    dataset_config: str,
    max_sequence_length: int,
    task_name: str,
) -> tuple:
    """
    Shared helper function to load and process TraitGym dataset.
    Returns: (df, reference_genome_path, flank_size)
    Note: Filtering by max_samples should be done by caller, not here.
    """
    
    reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
    flank_size = (max_sequence_length - 1) // 2

    if not os.path.exists(reference_genome_path):
        raise FileNotFoundError(
            f"Reference genome not found: {reference_genome_path}\n"
            f"Please download a reference genome (e.g., hg38.fa) from UCSC or Ensembl"
        )

    data_dir = os.path.join(root_data_dir_path, task_name)
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, f"TraitGym_{dataset_config}_data.parquet")

    if os.path.exists(parquet_path):
        logging.info(f"Loading TraitGym dataset from cache: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        logging.info("Downloading TraitGym dataset from songlab/TraitGym...")
        try:
            ds = load_dataset("songlab/TraitGym", dataset_config, split="test")
            df = ds.to_pandas()
            logging.info(f"Saving TraitGym dataset to: {parquet_path}")
            df.to_parquet(parquet_path, index=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download TraitGym dataset from songlab/TraitGym.\n"
                f"Error: {str(e)}\n"
                f"Please ensure the datasets library is installed and you have internet access."
            )

    logging.info(f"Loaded {len(df)} samples from TraitGym dataset")

    required_columns = ["chrom", "pos", "ref", "alt", "label"]
    if not all(col in df.columns for col in required_columns):
        available_cols = list(df.columns)
        raise ValueError(
            f"TraitGym dataset missing required columns. Expected {required_columns}, but found {available_cols}.\n"
            f"First few rows:\n{df.head()}"
        )

    logging.info(f"Loading reference genome: {reference_genome_path}")

    # Normalize dataframe columns
    df = df.copy()
    df['chrom'] = df['chrom'].astype(str).apply(lambda c: f"chr{c}" if not str(c).startswith('chr') else c)
    df['pos'] = df['pos'].astype(int)
    df['ref'] = df['ref'].astype(str).str.upper()
    df['alt'] = df['alt'].astype(str).str.upper()
    if df['label'].dtype == bool:
        df['label'] = df['label'].astype(int)

    logging.info("Filtering to keep only SNVs")
    snv_mask = (df['ref'].str.len() == 1) & (df['alt'].str.len() == 1)
    df = df[snv_mask]
    logging.info(f"After SNV filtering: {len(df)} samples")
    
    # Return df and metadata - caller will do sequence extraction and filtering
    return df, reference_genome_path, flank_size




class TraitGymMendelianTask(BaseGFMZeroShotSNVTask):
    """
    TraitGym mendelian_traits zero-shot SNV task.

    Loads TraitGym test split from HuggingFace (songlab/TraitGym) and extracts
    sequences around SNVs using the hg38 reference. The dataset is saved as
    a parquet cache at `data/TraitGym/TraitGym_{dataset_config}_data.parquet` after the first download.

    This task supports task_config options including:
    - max_num_samples: Limit number of samples for fast debugging
    - max_sequence_length: Maximum sequence length (default: 4096bp)
    - batch_size: Batch size for DataLoader (default: 32)
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize TraitGym task.
        
        Args:
            root_data_dir_path: Path to root data directory
            task_config: Optional configuration dictionary
            dataset_config: Dataset variant - "mendelian_traits" or "complex_traits"
        """
        # Set dataset-specific attributes before calling parent init
        self.dataset_config = "mendelian_traits"
        self.reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
        
        # Call parent initialization (handles task_config parsing and dataset creation)
        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        """Return task name (used for data directory)."""
        return "traitgym_mendelian"

    def _create_test_dataset(self) -> Dataset:
        """
        Create test dataset from TraitGym HuggingFace dataset.
        Extracts sequence contexts around each SNV from reference genome.
        Returns tuples of (variant_seq, reference_seq, label).
        """
        # Get dataframe and metadata from helper function
        df, reference_genome_path, flank_size = _load_traitgym_dataset(
            root_data_dir_path=self.root_data_dir_path,
            dataset_config=self.dataset_config,
            max_sequence_length=self.max_sequence_length,
            task_name=self.get_task_name(),
        )
        
        # Store flank_size for _get_variant_position_in_sequence
        self.flank_size = flank_size
        
        # Limit samples if max_num_samples is specified (follows BendVEPDisease pattern)
        if self.max_num_samples is not None:
            df = df.head(min(self.max_num_samples, len(df)))
            logging.info(f"Limiting to {len(df)} samples (max_num_samples={self.max_num_samples})")
        
        # Check reference genome exists
        if not os.path.exists(reference_genome_path):
            logging.info("Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(reference_genome_path)
        
        # Load reference genome
        logging.info(f"Loading reference genome: {reference_genome_path}")
        genome = Fasta(reference_genome_path)
        
        # Extract sequences from reference genome
        logging.info(f"Extracting sequences (window size: {self.max_sequence_length}bp, flank: {flank_size}bp)...")
        reference_sequences = []
        variant_sequences = []
        labels = []
        skipped = 0
        
        for chrom, group in df.groupby('chrom'):
            try:
                chrom_obj = genome[chrom]
            except KeyError:
                logging.warning(f"Chromosome {chrom} not found in reference genome")
                skipped += len(group)
                continue
            except Exception as e:
                logging.warning(f"Error accessing chromosome {chrom}: {e}")
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
            logging.info(f"Skipped {skipped} variants due to extraction issues or allele mismatch")

        logging.info(f"Successfully extracted {len(reference_sequences)} sequence pairs")
        
        # Store labels for metrics
        if len(labels) == 0:
            raise ValueError(
                "No valid samples after extracting sequences from reference genome. "
                "Check that hg38.fa is compatible with the TraitGym dataset."
            )
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Create dataset: (variant_sequence, reference_sequence, label, conditional_input) tuples
        test_dataset = [
            (var_seq, ref_seq, label, np.array([])) 
            for var_seq, ref_seq, label in zip(variant_sequences, reference_sequences, self.labels)
        ]
        
        return test_dataset

    def _get_variant_position_in_sequence(self) -> int:
        return self.flank_size

    def _get_default_max_seq_len(self) -> int:
        """Return the task's default maximum sequence length.

        TraitGym sequences can be large; use a conservative default (1048576bp).
        """
        return 1048576

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None



# def traitgym_mendelian(
#     root_data_dir_path: str,
#     task_config: Optional[Dict] = None,
# ) -> TraitGymTask:
#     """
#     Factory function to create a TraitGymTask instance for mendelian_traits (OMIM).
    
#     Args:
#         root_data_dir_path: Path to root data directory
#         task_config: Configuration dict with optional keys:
#             - max_sequence_length: Maximum sequence length (default: 4096bp)
#             - batch_size: Batch size for DataLoader (default: 32)
#             - max_num_samples: Maximum number of samples to use (default: None, use all)
    
#     Returns:
#         TraitGymTask instance configured for mendelian_traits dataset
#     """
#     return TraitGymTask(
#         root_data_dir_path=root_data_dir_path,
#         task_config=task_config,
#         dataset_config="mendelian_traits",
#     )


# def traitgym_complex(
#     root_data_dir_path: str,
#     task_config: Optional[Dict] = None,
# ) -> TraitGymTask:
#     """
#     Factory function to create a TraitGymTask instance for complex_traits.
    
#     Args:
#         root_data_dir_path: Path to root data directory
#         task_config: Configuration dict with optional keys:
#             - max_sequence_length: Maximum sequence length (default: 4096bp)
#             - batch_size: Batch size for DataLoader (default: 32)
#             - max_num_samples: Maximum number of samples to use (default: None, use all)
    
#     Returns:
#         TraitGymTask instance configured for complex_traits dataset
#     """
#     return TraitGymTask(
#         root_data_dir_path=root_data_dir_path,
#         task_config=task_config,
#         dataset_config="complex_traits",
#     )