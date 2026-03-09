import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pyfaidx import Fasta
from huggingface_hub import hf_hub_download

from gfmbench_api.utils.fileutils import ensure_reference_genome
from gfmbench_api.utils.preprocutils import standardize_sequence, pad_sequence_centered_variant
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask



# Dataset Logic 

class _PathogenicOmimDataset(Dataset):
    """
    Dataset that returns (variant_sequence, reference_sequence, label, conditional_input).
    conditional_input is None for this task (no metadata).
    """
    def __init__(self, df: pd.DataFrame, fasta: Fasta, seq_len: int):
        self.fasta = fasta
        self.seq_len = seq_len
        self.center = seq_len // 2
        
        # Filter Test Split (if column exists)
        if "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == "test"]
        
        # Filter for SNVs only
        df = df[df["ALT"].astype(str).str.len() == 1]
        df = df[df["ALT"].isin(["A", "C", "G", "T"])]
        
        # Validate Boundaries
        valid_indices = []
        logging.info(f"Pre-filtering {len(df)} samples for validity...")
        
        for idx in df.index:
            try:
                chrom = str(df.at[idx, "CHROM"])
                pos = int(df.at[idx, "POS"])
                
                # Check chromosome existence
                if chrom not in fasta.keys():
                    continue
                    
                chrom_len = len(fasta[chrom])
                start = (pos - 1) - self.center
                end = start + self.seq_len
                
                # Check boundaries
                if start >= 0 and end <= chrom_len:
                    valid_indices.append(idx)
            except Exception:
                continue
                
        self.df = df.loc[valid_indices].reset_index(drop=True)
        logging.info(f"Kept {len(self.df)} valid samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        chrom = str(row["CHROM"])
        pos1 = int(row["POS"])
        alt = str(row["ALT"]).upper()
        y = int(row["INT_LABEL"])
        
        pos_0 = pos1 - 1
        
        # Get Reference Sequence using padding function (handles chromosome boundaries)
        ref_seq_raw = pad_sequence_centered_variant(
            chromosome=self.fasta[chrom],
            variant_pos_0based=pos_0,
            max_sequence_length=self.seq_len,
            variant_pos_in_seq=self.center
        )
        ref_seq = standardize_sequence(ref_seq_raw)
        
        # Create Variant Sequence (Mutate Center as mentioned in the LRB paper)
        alt_seq_list = list(ref_seq)
        alt_seq_list[self.center] = alt
        alt_seq = standardize_sequence("".join(alt_seq_list))
        
        # Return (variant_seq, ref_seq, label, conditional_input)
        # conditional_input is empty array for this task (no metadata)
        return alt_seq, ref_seq, y, np.array([])



# Task Class 
class LrbVariantEffectPathogenicOmimTask(BaseGFMZeroShotSNVTask):
    """
    LRB Pathogenic OMIM Task.
    """

    def __init__(self, root_data_dir_path, task_config=None):
        self.reference_genome_path = os.path.join(root_data_dir_path, "reference_genome", "hg38.fa")
        super().__init__(root_data_dir_path, task_config)

    def get_task_name(self) -> str:
        return "lrb_variant_effect_pathogenic_omim"

    def _get_default_max_seq_len(self) -> int:
        return 1048576

    def _get_variant_position_in_sequence(self) -> int:
        return self.max_sequence_length // 2

    def _create_test_dataset(self) -> Dataset:
        cfg = self.task_config or {}
        
        # Setup Paths based on Root Dir
        task_data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        os.makedirs(task_data_dir, exist_ok=True)
        
        # Ensure reference genome exists (auto-download if missing)
        if not os.path.exists(self.reference_genome_path):
            logging.info(f"Reference genome not found. Downloading hg38.fa...")
            ensure_reference_genome(self.reference_genome_path)
        
        # Ensure/Load Variants Data
        variants_path = cfg.get("variants_path")
        
        if not variants_path:
            expected_filename = "vep_pathogenic_non_coding_subset.csv"
            flat_path = os.path.join(task_data_dir, expected_filename)
            
            if os.path.exists(flat_path):
                variants_path = flat_path
            else:
                logging.info(f"[Task] Data not found. Downloading from InstaDeepAI/genomics-long-range-benchmark...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id="InstaDeepAI/genomics-long-range-benchmark",
                        filename="variant_effect_pathogenic/vep_pathogenic_non_coding_subset.csv",
                        repo_type="dataset",
                        local_dir=task_data_dir
                    )
                    variants_path = downloaded_path
                except Exception as e:
                    raise RuntimeError(f"Failed to download LRB data: {e}")
        
        logging.info(f"[Task] Using Variants File: {variants_path}")
        df = pd.read_csv(variants_path)

        # Max Num Samples implementation
        max_samples = cfg.get("max_num_samples")
        if max_samples is not None:
            if len(df) > max_samples:
                logging.info(f"[Task] Applying max_num_samples limit: {max_samples} (Randomly sampled)")
                # shuffle first to avoid getting only Positives (since file is sorted)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                df = df.head(max_samples)

        # Initialize Dataset 
        context_len = cfg.get("max_sequence_length", self._get_default_max_seq_len())
        self.max_sequence_length = context_len 
        
        logging.info(f"[Task] Loading Genome: {self.reference_genome_path}")
        fasta = Fasta(str(self.reference_genome_path), one_based_attributes=False)
        
        return _PathogenicOmimDataset(df, fasta, context_len)

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None