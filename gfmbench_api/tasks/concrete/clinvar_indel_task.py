# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz — LicenseRef-NCBI-Data
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
"""
ClinVar Indel zero-shot task for insertions and deletions.

Evaluates genomic language models on pathogenicity prediction of ClinVar indel variants.
Downloads and processes raw ClinVar data if the benchmark parquet doesn't exist.
"""
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from gfmbench_api.tasks.base.base_gfm_zeroshot_general_indel_task import BaseGFMZeroShotGeneralIndelTask
from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant

# ClinVar annotation mappings
REVIEW_STATUS_TO_GOLD_STARS = {
    'criteria provided, single submitter': 1,
    'criteria provided, multiple submitters, no conflicts': 2,
    'criteria provided, conflicting interpretations': 1,
    'no assertion criteria provided': np.nan,
    'reviewed by expert panel': 3,
    'no assertion provided': np.nan,
    'no interpretation for the single variant': np.nan,
    'practice guideline': 4,
}

CLINICAL_SIGNIFICANCE_TO_LABEL = {
    'Benign': 0,
    'Likely benign': 0,  # Collapsed to binary
    'Benign/Likely benign': 0,
    'Pathogenic': 1,
    'Likely pathogenic': 1,  # Collapsed to binary
    'Pathogenic/Likely pathogenic': 1,
}


class IndelClinvarTask(BaseGFMZeroShotGeneralIndelTask):

    """
    ClinVar Indel zero-shot task for insertions and deletions.

    Evaluates variant effect prediction on ~13K high-confidence ClinVar indels
    (2+ gold stars, clear benign/pathogenic annotations, GRCh38).
    Includes both germline and somatic variants.

    For indels, ref and alt sequences have different lengths:
    - Deletions: alt sequence is shorter (bases removed)
    - Insertions: alt sequence is longer (bases added)

    Data is auto-downloaded and processed from NCBI ClinVar if not present.

    Config options (via task_config):
    - max_sequence_length: Context window size (default: 512bp via config, or 1048576bp if not specified)
    - max_num_samples: Limit samples for fast testing (default: None)
    """

    def __init__(
        self,
        root_data_dir_path: str,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        root_data_dir_path = os.path.expanduser(root_data_dir_path)
        # ClinVar data source
        self.clinvar_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-01.txt.gz"
        self.indel_types = ['Insertion', 'Deletion', 'Indel']
        # Reference genome path
        self.reference_genome_path = os.path.join(
            root_data_dir_path, "reference_genome", "hg38.fa"
        )

        # Data directory for this task (follows framework convention)
        self._task_data_dir = os.path.join(root_data_dir_path, "clinvar_indel")

        # Paths for raw and processed data
        self._raw_data_path = os.path.join(self._task_data_dir, "variant_summary_2026-01.txt.gz")
        self._parquet_path = os.path.join(self._task_data_dir, "clinvar_indel_benchmark.parquet")

        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        return "clinvar_indel"

    def _get_default_max_seq_len(self) -> int:
        """Default context window: ~1M , 2^20."""
        return 1048576

    def _get_variant_position_in_sequence(self) -> int:
        """
        Return the nominal position of the variant start in the reference sequence.
        
        For indels, this is the center of the context window. The actual variant
        may span multiple positions (for deletions) or the alt may extend beyond
        this position (for insertions).

        Examples:
            >>> # Typical even length
            >>> task = ClinvarIndelTask.__new__(ClinvarIndelTask)
            >>> task.max_sequence_length = 1000
            >>> task._get_variant_position_in_sequence()
            500

            >>> # Odd length (floor division)
            >>> task.max_sequence_length = 1001
            >>> task._get_variant_position_in_sequence()
            500
        """
        return self.max_sequence_length // 2

    # =========================================================================
    # Data Preparation Functions
    # =========================================================================

    def _download_clinvar_raw(self) -> None:
        """Download raw ClinVar variant_summary.txt.gz from NCBI."""
        os.makedirs(self._task_data_dir, exist_ok=True)

        if os.path.exists(self._raw_data_path):
            logging.info(f"Raw ClinVar data exists: {self._raw_data_path}")
            return

        logging.info("Downloading ClinVar data from NCBI (~500MB)...")
        try:
            result = subprocess.run(
                ["wget", "-q", "-O", self._raw_data_path, self.clinvar_url],
                capture_output=True, text=True, timeout=900
            )
            if result.returncode != 0:
                raise RuntimeError(f"wget failed: {result.stderr}")
            logging.info(f"Downloaded to: {self._raw_data_path}")
        except FileNotFoundError:
            # wget not available, try urllib
            import urllib.request
            logging.info("wget not found, using urllib...")
            urllib.request.urlretrieve(self.clinvar_url, self._raw_data_path)
            logging.info(f"Downloaded to: {self._raw_data_path}")

    def _filter_and_prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ClinVar data to high-quality indel variants.

        Filtering steps:
        1. GRCh38 assembly only
        2. Standard chromosomes (1-22, X, Y, MT)
        3. Indel types only (Insertion, Deletion, Indel)
        4. Clear pathogenicity (Benign or Pathogenic, not VUS)
        5. High confidence (2+ gold stars)

        Note: All variant origins included (germline, somatic, etc).
        Final sample count may differ from parquet due to ref allele mismatches
        during sequence extraction from the reference genome (~1-2% typical).

        Returns:
            DataFrame with columns: chrom, pos, ref, alt, label, variant_type
        """
        n_orig = len(df)
        logging.info(f"Filtering {n_orig:,} raw ClinVar variants...")

        # GRCh38 only
        df = df[df['Assembly'] == 'GRCh38'].copy()
        logging.info(f"  GRCh38: {len(df):,}")

        # # Germline (exclude somatic)
        # df = df[df['OriginSimple'] != 'somatic'].copy()
        # logging.info(f"  Germline: {len(df):,}")

        # Standard chromosomes
        standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        df = df[df['Chromosome'].isin(standard_chroms)].copy()
        logging.info(f"  Standard chromosomes: {len(df):,}")

        # Indel types only
        if 'Type' not in df.columns:
            raise ValueError("ClinVar data missing 'Type' column")
        df = df[df['Type'].isin(self.indel_types)].copy()
        logging.info(f"  Indel types: {len(df):,}")

        # Map clinical significance to binary labels
        df['label'] = df['ClinicalSignificance'].map(CLINICAL_SIGNIFICANCE_TO_LABEL)
        df = df[df['label'].isin({0, 1})].copy()
        logging.info(f"  Benign/Pathogenic only: {len(df):,}")

        # Map review status to gold stars
        df['gold_stars'] = df['ReviewStatus'].map(REVIEW_STATUS_TO_GOLD_STARS)
        df = df[df['gold_stars'] >= 2].copy()
        logging.info(f"  2+ gold stars: {len(df):,}")

        # Prepare output columns
        result = pd.DataFrame({
            'chrom': df['Chromosome'].astype(str).apply(
                lambda c: f"chr{c}" if not c.startswith('chr') else c
            ),
            'pos': df['PositionVCF'].astype(int),
            'ref': df['ReferenceAlleleVCF'].astype(str).str.upper(),
            'alt': df['AlternateAlleleVCF'].astype(str).str.upper(),
            'label': df['label'].astype(int),
            'variant_type': df['Type'].astype(str),
        })

        # Drop rows with missing values
        result = result.dropna().reset_index(drop=True)
        logging.info(f"Final dataset: {len(result):,} indel variants")

        return result

    def _ensure_parquet_exists(self) -> None:
        """Ensure the processed parquet file exists, creating it if needed."""
        if os.path.exists(self._parquet_path):
            return

        logging.info("Parquet not found. Processing ClinVar data...")

        # Download raw data if needed
        self._download_clinvar_raw()

        # Load and filter
        logging.info(f"Loading raw data: {self._raw_data_path}")
        df = pd.read_csv(
            self._raw_data_path, sep='\t', compression='gzip',
            dtype={'Chromosome': str}, low_memory=False
        )

        # Filter and prepare
        df = self._filter_and_prepare_dataframe(df)

        # Save
        df.to_parquet(self._parquet_path, index=False)
        logging.info(f"Saved to: {self._parquet_path}")

    # =========================================================================
    # Sequence Extraction
    # =========================================================================

    def _extract_sequences(
        self, df: pd.DataFrame, genome
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Extract ref and alt sequences from reference genome for each variant.

        For indels:
        - ref_seq: context window from reference genome (contains ref allele)
        - alt_seq: context with ref allele replaced by alt allele

        Variants are skipped if:
        - Chromosome not found in reference genome
        - Reference allele doesn't match the genome (indicates misalignment)

        Returns:
            (variant_seqs, reference_seqs, labels, variant_positions)

        Key behaviors tested in _test_indel_sequence_extraction():
        - Position conversion: 1-based VCF to 0-based
        - Deletions: len(alt) < len(ref) → alt_seq shorter
        - Insertions: len(alt) > len(ref) → alt_seq longer
        - Reference validation uses full ref_allele length (not just 1bp)
        - Window centering at variant position
        - Chromosome boundary handling
        """
        half_window = self.max_sequence_length // 2

        reference_sequences = []
        variant_sequences = []
        labels = []
        variant_positions = []  # Position of variant start in ref_seq
        skipped = 0

        # Process per chromosome for efficiency
        for chrom, group in df.groupby("chrom"):
            try:
                chrom_obj = genome[chrom]
            except KeyError:
                logging.warning(f"Chromosome {chrom} not in reference, skipping {len(group)} variants")
                skipped += len(group)
                continue

            for _, row in group.iterrows():
                pos = int(row["pos"])  # 1-based VCF position
                ref_allele = str(row["ref"]).upper()
                alt_allele = str(row["alt"]).upper()
                label = int(row["label"])

                # Convert to 0-based
                pos_0 = pos - 1

                # Extract reference sequence using padding function (handles chromosome boundaries)
                try:
                    ref_seq = pad_sequence_centered_variant(
                        chromosome=chrom_obj,
                        variant_pos_0based=pos_0,
                        max_sequence_length=self.max_sequence_length,
                        variant_pos_in_seq=half_window
                    )
                except Exception as e:
                    logging.debug(f"Error extracting sequence for {chrom}:{pos}. {str(e)}. Skipping.")
                    skipped += 1
                    continue

                # Validate sequence length
                if len(ref_seq) != self.max_sequence_length:
                    skipped += 1
                    continue

                # Variant position in window should be at half_window (center)
                variant_pos_in_window = half_window
                
                # Check if padding character is at variant position (shouldn't happen, but safety check)
                if variant_pos_in_window < len(ref_seq) and 'P' in ref_seq[variant_pos_in_window:variant_pos_in_window + len(ref_allele)]:
                    logging.debug(f"Variant position at {chrom}:{pos} falls in padding region. Skipping.")
                    skipped += 1
                    continue

                # Validate reference allele matches genome
                extracted_ref = ref_seq[variant_pos_in_window:variant_pos_in_window + len(ref_allele)]

                if extracted_ref != ref_allele:
                    logging.debug(f"Ref mismatch at {chrom}:{pos}")
                    skipped += 1
                    continue

                # alt_seq: context with ref allele replaced by alt allele
                # This naturally handles length changes:
                # - Deletion: len(alt) < len(ref) -> alt_seq shorter
                # - Insertion: len(alt) > len(ref) -> alt_seq longer
                alt_seq = (
                    ref_seq[:variant_pos_in_window] +
                    alt_allele +
                    ref_seq[variant_pos_in_window + len(ref_allele):]
                )

                reference_sequences.append(ref_seq)
                variant_sequences.append(alt_seq)
                labels.append(label)
                variant_positions.append(variant_pos_in_window)

        if skipped > 0:
            logging.info(f"Skipped {skipped} variants (ref mismatch or missing chrom)")

        return variant_sequences, reference_sequences, labels, variant_positions

    # =========================================================================
    # Main Dataset Creation
    # =========================================================================

    def _create_test_dataset(self):
        """
        Create test dataset: (variant_seq, reference_seq, label, cond_input) tuples.
        
        Downloads and processes ClinVar data if the parquet doesn't exist.
        Extracts sequence contexts from the hg38 reference genome.
        """
        try:
            from pyfaidx import Fasta
        except ImportError:
            raise ImportError(
                "pyfaidx required: pip install pyfaidx"
            )

        # Ensure reference genome
        if not os.path.exists(self.reference_genome_path):
            logging.info("Downloading reference genome...")
            ensure_reference_genome(self.reference_genome_path)

        # Ensure parquet (downloads and processes if needed)
        self._ensure_parquet_exists()

        # Load parquet
        logging.info(f"Loading: {self._parquet_path}")
        df = pd.read_parquet(self._parquet_path)
        logging.info(f"Loaded {len(df):,} indel variants")

        # Early slice for fast testing
        if self.max_num_samples is not None:
            df = df.iloc[:self.max_num_samples]
            logging.info(f"[Fast run] Using first {len(df)} variants")

        # Load reference genome
        logging.info("Loading reference genome...")
        genome = Fasta(self.reference_genome_path)

        # Extract sequences
        logging.info(f"Extracting sequences (window={self.max_sequence_length}bp)...")
        variant_seqs, ref_seqs, labels, var_positions = self._extract_sequences(df, genome)

        if len(labels) == 0:
            raise ValueError("No valid samples after sequence extraction")

        # Store for potential analysis
        self.labels = torch.tensor(labels, dtype=torch.long)
        self._variant_positions = var_positions  # For validation/debugging

        # Create dataset tuples: (variant_seq, ref_seq, label, conditional_input)
        test_dataset = [
            (var_seq, ref_seq, label, np.array([], dtype=np.float32))
            for var_seq, ref_seq, label in zip(variant_seqs, ref_seqs, self.labels)
        ]

        # Log summary
        n_ins = sum(1 for v, r, *_ in test_dataset if len(v) > len(r))
        n_del = sum(1 for v, r, *_ in test_dataset if len(v) < len(r))
        n_path = sum(1 for *_, lbl, _ in test_dataset if lbl == 1)
        logging.info(
            f"Dataset ready: {len(test_dataset)} samples "
            f"({n_ins} insertions, {n_del} deletions, "
            f"{n_path} pathogenic, {len(test_dataset)-n_path} benign)"
        )

        return test_dataset

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """No conditional metadata for this task."""
        return None