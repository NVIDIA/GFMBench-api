# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_disease.bed — BSD-3-Clause
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch

from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
import numpy as np
from gfmbench_api.utils.fileutils import download_file_from_url, ensure_reference_genome
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant


class BendVEPDisease(BaseGFMZeroShotSNVTask):
    """
    Variant Effect Prediction (VEP) task for disease variants from BEND dataset.
    Uses Log-Likelihood Ratio (LLR) to evaluate zero-shot variant effect prediction.
    
    Extracts sequence contexts around each variant from a reference genome.
    Requires a reference genome FASTA file (e.g., hg38.fa).
    """
    
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VEP Disease task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """
        # Data source URL (BEND repository)
        self.data_source_url = "https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_disease.bed"
        
        # Set reference genome path
        self.reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
        
        # Call parent initialization (computes self.max_sequence_length and calls _create_datasets)
        super().__init__(root_data_dir_path, task_config)
    
    def _get_default_max_seq_len(self) -> int:
        """Return task's default maximum sequence length (1048576bp)."""
        return 512
    
    
    def get_task_name(self):
        """Return task name (identical to data directory name)."""
        return "bend_variant_effects_disease"
    
    def _create_test_dataset(self):
        """
        Create test dataset from BED file.
        Extracts sequence contexts around each variant from reference genome.
        Returns tuples of (variant_seq, reference_seq, label).
        """
        # Import pyfaidx for reading reference genome
        try:
            from pyfaidx import Fasta
        except ImportError:
            raise ImportError(
                "pyfaidx is required for extracting sequences from reference genome.\n"
                "Install with: pip install pyfaidx"
            )
        
        # Ensure reference genome exists (auto-download if missing)
        if not os.path.exists(self.reference_genome_path):
            logging.info(f"Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)
        
        # Load reference genome
        logging.info(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)
        
        # Load BED file
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        data_path = os.path.join(data_dir, "data.bed")
        
        # Download data if not exists
        if not os.path.exists(data_path):
            logging.info(f"Downloading {self.get_task_name()} from BEND repository...")
            download_file_from_url(self.data_source_url, data_path)
            logging.info(f"Data saved to: {data_dir}")
        
        # Load BED file (tab-separated)
        df = pd.read_csv(data_path, sep='\t')
        
        # Limit samples if max_num_samples is specified
        if self.max_num_samples is not None:
            df = df.head(min(self.max_num_samples, len(df)))
        
        # Extract sequences from reference genome
        logging.info(f"Extracting sequences (window size: {self.max_sequence_length}bp)...")
        reference_sequences = []
        variant_sequences = []
        labels = []
        full_labels = []
        chromosomes = []
        positions = []
        
        for idx, row in df.iterrows():
            chrom = str(row['chromosome'])
            pos = int(row['start'])  # Variant position (0-based in BED format)
            ref_allele = str(row['ref']).upper()
            alt_allele = str(row['alt']).upper()
            
            # Get variant position in sequence (center)
            variant_pos_in_seq = self._get_variant_position_in_sequence()
            
            # Extract reference sequence using padding function (handles chromosome boundaries)
            try:
                ref_seq = pad_sequence_centered_variant(
                    chromosome=genome[chrom],
                    variant_pos_0based=pos,
                    max_sequence_length=self.max_sequence_length,
                    variant_pos_in_seq=variant_pos_in_seq
                )
            except KeyError:
                logging.warning(f"Chromosome {chrom} not found in reference genome. Skipping variant at position {pos}")
                continue
            except Exception as e:
                logging.warning(f"Error extracting sequence for {chrom}:{pos}. {str(e)}. Skipping.")
                continue
            
            # Verify sequence length
            if len(ref_seq) != self.max_sequence_length:
                logging.warning(f"Sequence length mismatch at {chrom}:{pos}. Expected {self.max_sequence_length}, got {len(ref_seq)}. Skipping.")
                continue
            
            # Assert that the reference allele matches the genome at variant position
            if variant_pos_in_seq < len(ref_seq):
                extracted_ref = ref_seq[variant_pos_in_seq:variant_pos_in_seq + len(ref_allele)]
                # Skip if padding character is at variant position (shouldn't happen, but safety check)
                if 'P' in extracted_ref:
                    logging.warning(f"Variant position at {chrom}:{pos} falls in padding region. Skipping.")
                    continue
                assert extracted_ref == ref_allele, (
                    f"Reference nucleotide mismatch at {chrom}:{pos}. "
                    f"Dataframe has '{ref_allele}' but reference genome has '{extracted_ref}'. "
                    f"This indicates the dataframe and reference genome are incompatible."
                )
            else:
                raise ValueError(
                    f"Variant position {variant_pos_in_seq} is out of bounds for sequence of length {len(ref_seq)} "
                    f"at {chrom}:{pos}"
                )
            
            # Create variant sequence by substituting the allele
            var_seq = (ref_seq[:variant_pos_in_seq] + 
                      alt_allele + 
                      ref_seq[variant_pos_in_seq + len(ref_allele):])
            
            reference_sequences.append(ref_seq)
            variant_sequences.append(var_seq)
            
            # Store metadata for valid samples only
            labels.append(row['label'])
            full_labels.append(row['full_label'])  # Disease dataset has 'full_label' column
            chromosomes.append(row['chromosome'])
            positions.append(row['start'])
        
        logging.info(f"Successfully extracted {len(reference_sequences)} sequence pairs")
        
        # Store labels and metadata for valid samples
        if len(labels) == 0:
            raise ValueError(
                f"No labels found in the BED file. The 'label' column is required for zero-shot tasks."
            )
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.full_labels = full_labels if len(full_labels) > 0 else None
        self.chromosomes = chromosomes if len(chromosomes) > 0 else None
        self.positions = positions if len(positions) > 0 else None
        
        # Create dataset: (variant_sequence, reference_sequence, label, conditional_input) tuples
        test_dataset = [
            (var_seq, ref_seq, label, np.array([])) 
            for var_seq, ref_seq, label in zip(variant_sequences, reference_sequences, self.labels)
        ]
        
        return test_dataset

    def _get_variant_position_in_sequence(self):
        """Return the position of the variant in the sequence."""
        return self.max_sequence_length // 2

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None



