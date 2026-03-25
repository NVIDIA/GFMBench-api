# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx — Apache-2.0
# - https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/GRCh37.p13_chr17.fna.gz — Apache-2.0
import os
from typing import Any, Dict, Optional, List
import pandas as pd
import torch
from gfmbench_api.tasks.base.base_gfm_zeroshot_snv_task import BaseGFMZeroShotSNVTask
from gfmbench_api.metrics import SNVVariantEffectPredictionLLRAUROC
from gfmbench_api.utils.fileutils import download_file_from_url
from gfmbench_api.utils.preprocutils import pad_sequence_centered_variant
import glob
import json
import subprocess
from pathlib import Path
import gzip

import numpy as np
import torch


class BRCA1Task(BaseGFMZeroShotSNVTask):
    """
    Zero-shot prediction of BRCA1 variant effects from a DMS study. 
    BRCA1 gene encodes for a protein that repairs damaged DNA (Moynahan et al., 1999). We predict whether a particular single nucleotide variant (SNV) of the BRCA1 gene is likely to be harmful to the protein's function, and thus potentially increase the risk of cancer for the patient with the genetic variant.

    We start by loading a dataset from Findlay et al. (2018), which contains experimentally measured function scores of 3,893 BRCA1 SNVs. These function scores reflect the extent by which the genetic variant has disrupted the protein's function, with lower scores indicating greater disruption. In this dataset, the SNVs are classified into three categories based on their function scores: LOF (loss-of-function), INT (intermediate), and FUNC (functional). We group Func/Int and do 0 shot classification here. 
    Uses Log-Likelihood Ratio (LLR) to evaluate zero-shot variant effect prediction.
    
    Based on: https://github.com/ArcInstitute/evo2/blob/main/notebooks/brca1/brca1_zero_shot_vep.ipynb

    Extracts sequence contexts around each variant from a reference genome.
    Requires a reference genome FASTA file (e.g., hg38.fa).

    """
    
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BRCA1 Zero-Class task.
        """
        # --- CHANGE 1: Point to the specific downloaded reference file ---
        # It is located in the task folder (brca1), not the shared reference_genome folder
        self.reference_genome_path = os.path.join(root_data_dir_path, self.get_task_name(), "GRCh37.p13_chr17.fna")

        cfg = task_config or {}
        self.max_samples = cfg.get("max_samples", None)

        super().__init__(root_data_dir_path, task_config)
    
    def _get_default_max_seq_len(self) -> int:
        return 1048576
    
    def get_task_name(self):
        return "brca1" 
    
    def _create_test_dataset(self):
        try:
            from pyfaidx import Fasta
        except ImportError:
            raise ImportError("pyfaidx is required. Install with: pip install pyfaidx")
        
        data_dir = os.path.join(self.root_data_dir_path, self.get_task_name())
        data_path = os.path.join(data_dir, "brca1.parquet")
        
        # Auto-download dataset and reference genome if missing (consistent with ClinVar task pattern)
        if not os.path.exists(data_path) or not os.path.exists(self.reference_genome_path):
            print("BRCA1 dataset or reference genome not found. Downloading...")
            try:
                self.load_save_brca1_dataset(output_data_dir_path=data_dir)
                print(f"Successfully downloaded BRCA1 dataset and reference to {data_dir}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download BRCA1 dataset and reference genome.\n"
                    f"Error: {str(e)}"
                ) from e
        
        if not os.path.exists(self.reference_genome_path):
            raise FileNotFoundError(
                f"Reference genome not found after download attempt: {self.reference_genome_path}"
            )
        
        print(f"Loading reference genome: {self.reference_genome_path}")
        genome = Fasta(self.reference_genome_path)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found after download attempt: {data_path}")
        
        df = pd.read_parquet(data_path)
        
        # Ensure chrom is "chr17"
        df['chrom'] = df['chrom'].astype(str).apply(lambda c: f"chr{c}" if not str(c).startswith('chr') else c)
        
        print(f"Extracting sequences (window size: {self.max_sequence_length}bp)...")
        reference_sequences = []
        variant_sequences = []
        labels = []
        chromosomes = []
        positions = []
        
        for idx, row in df.iterrows():
            if self.max_samples is not None and len(reference_sequences) >= self.max_samples:
                print(f"[Fast run] Using only first {self.max_samples} samples.")
                break
                
            chrom = str(row['chrom'])
            
            # --- CHANGE 2: Fix Indexing Bug ---
            # Data is 1-based (hg19), Python is 0-based. Must subtract 1.
            pos = int(row['pos']) - 1 
            
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
                print(f"Warning: {chrom} not found in genome keys {list(genome.keys())}. Skipping.")
                continue
            except Exception as e:
                print(f"Error extracting sequence for {chrom}:{pos+1}. {str(e)}. Skipping.")
                continue
            
            # Verify sequence length
            if len(ref_seq) != self.max_sequence_length:
                print(f"Sequence length mismatch at {chrom}:{pos+1}. Expected {self.max_sequence_length}, got {len(ref_seq)}. Skipping.")
                continue
            
            # Assert that the reference allele matches the genome at variant position
            if variant_pos_in_seq < len(ref_seq):
                extracted_ref = ref_seq[variant_pos_in_seq:variant_pos_in_seq + len(ref_allele)]
                # Skip if padding character is at variant position (shouldn't happen, but safety check)
                if 'P' in extracted_ref:
                    print(f"Variant position at {chrom}:{pos+1} falls in padding region. Skipping.")
                    continue
                assert extracted_ref == ref_allele, (
                    f"Reference mismatch at {chrom}:{pos+1}. "
                    f"Data: '{ref_allele}', Genome: '{extracted_ref}'."
                )
            else:
                continue
            
            var_seq = (ref_seq[:variant_pos_in_seq] + 
                      alt_allele + 
                      ref_seq[variant_pos_in_seq + len(ref_allele):])
            
            reference_sequences.append(ref_seq)
            variant_sequences.append(var_seq)
            labels.append(1 - row['label'])
            chromosomes.append(row['chrom'])
            
            # 'start' does not exist in the dataframe, use 'pos' (storing original 1-based pos is standard)
            positions.append(row['pos']) 
        
        print(f"Successfully extracted {len(reference_sequences)} sequence pairs")
        
        if len(labels) == 0:
            raise ValueError("No valid samples found.")
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.chromosomes = chromosomes
        self.positions = positions
        
        test_dataset = [
            (var_seq, ref_seq, label, np.array([])) 
            for var_seq, ref_seq, label in zip(variant_sequences, reference_sequences, self.labels)
        ]
        
        return test_dataset

    def _get_variant_position_in_sequence(self):
        return self.max_sequence_length // 2
    
    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
    
    def load_save_brca1_dataset(self, output_data_dir_path: str):
        """
        Load and save dataset as parquet, in VCF like format, for standardized loading (as 0 shot) later. 
        ADDITIONALLY Download, save hg19 - chrom 17 fasta+gzip file! (it is not hg38 like rest of our datasets - and reference is based on that)

        Zero-shot prediction of BRCA1 variant effects from a DMS study. 
        BRCA1 gene encodes for a protein that repairs damaged DNA (Moynahan et al., 1999). We predict whether a particular single nucleotide variant (SNV) of the BRCA1 gene is likely to be harmful to the protein's function, and thus potentially increase the risk of cancer for the patient with the genetic variant.

        We start by loading a dataset from Findlay et al. (2018), which contains experimentally measured function scores of 3,893 BRCA1 SNVs. These function scores reflect the extent by which the genetic variant has disrupted the protein's function, with lower scores indicating greater disruption. In this dataset, the SNVs are classified into three categories based on their function scores: LOF (loss-of-function), INT (intermediate), and FUNC (functional). We group Func/Int

        Based on: https://github.com/ArcInstitute/evo2/blob/main/notebooks/brca1/brca1_zero_shot_vep.ipynb

        :param output_data_dir_path: Directory path where the dataset and reference genome will be saved
        :type output_data_dir_path: str
        """
        os.makedirs(output_data_dir_path, exist_ok=True)
        
        # 1. Download and Process BRCA1 Dataset
        data_path = os.path.join(output_data_dir_path, "brca1.parquet")
        if not os.path.exists(data_path):
            print("Downloading and processing BRCA1 dataset...")
            df = pd.read_excel("https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx", engine='openpyxl', header=2)
            df = df[[ 'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class', ]]
            
            df.rename(columns={
                'chromosome': 'chrom',
                'position (hg19)': 'pos',
                'reference': 'ref',
                'alt': 'alt',
                'function.score.mean': 'score',
                'func.class': 'class',
            }, inplace=True)

            # Convert to two-class system
            df['class'] = df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')
            df['label'] = df['class'].apply(lambda x: 1 if x == 'FUNC/INT' else 0 if x == "LOF" else np.nan)
            df.drop(columns=['class'], inplace=True)
            print(df.head())
            assert df['label'].isnull().sum() == 0, "All classes should be mapped to labels"
            assert df["label"].nunique() >1, "There should be exactly two classes after mapping"
            df.to_parquet(data_path, index=False)
            print(f"Saved BRCA1 DMS dataset with {len(df)} variants to parquet.")
        else:
            print(f"BRCA1 dataset already exists at {data_path}")

        # 2. Download and Prepare Reference Genome
        fasta_url = "https://github.com/ArcInstitute/evo2/raw/refs/heads/main/notebooks/brca1/GRCh37.p13_chr17.fna.gz"
        local_fna_path = os.path.join(output_data_dir_path, "GRCh37.p13_chr17.fna")
        
        if not os.path.exists(local_fna_path):
            print(f"Downloading reference genome...")
            local_gz_path = os.path.join(output_data_dir_path, "GRCh37.p13_chr17.fna.gz")
            
            try:
                # Use the centralized download utility
                download_file_from_url(fasta_url, local_gz_path)
                    
                print("Unzipping and normalizing FASTA header...")
                # We rewrite the header to '>chr17' so pyfaidx can find it easily
                with gzip.open(local_gz_path, 'rt') as f_in, open(local_fna_path, 'w') as f_out:
                    first_line = True
                    for line in f_in:
                        if first_line and line.startswith('>'):
                            # Force the header to be >chr17 to match dataframe and task logic
                            f_out.write(">chr17\n") 
                            first_line = False
                        else:
                            f_out.write(line)
                
                os.remove(local_gz_path)  # Cleanup .gz file
                print(f"Reference genome ready: {local_fna_path}")
                
            except Exception as e:
                print(f"Error downloading or processing FASTA: {e}")
                raise
        else:
            print(f"Reference genome already exists at {local_fna_path}")


def _write_dedup_fasta(seqs, fasta_path: Path, prefix: str):
    """
    Writes a FASTA containing each unique sequence exactly once.
    Returns:
      - names_in_order: list[str] aligned with input seqs
      - seq_to_name: dict[str, str] for debugging
    """
    seq_to_name = {}
    entries = []
    names_in_order = []

    for seq in seqs:
        if seq not in seq_to_name:
            name = f"{prefix}_{len(seq_to_name)}"
            seq_to_name[seq] = name
            entries.append(f">{name}\n{seq}\n")
        names_in_order.append(seq_to_name[seq])

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text("".join(entries))
    return names_in_order, seq_to_name


def _load_predictions(pred_dir: Path):
    """
    Loads:
      - seq_idx_map.json (name -> idx)
      - predictions__rank_*.pt (contains 'log_probs_seqs')
    """
    with open(pred_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(f)

    pred_files = sorted(glob.glob(str(pred_dir / "predictions__rank_*.pt")))
    if not pred_files:
        raise FileNotFoundError(f"No predictions__rank_*.pt found under {pred_dir}")
    preds = torch.load(pred_files[0], map_location="cpu")

    if "log_probs_seqs" not in preds:
        raise KeyError(f"Expected 'log_probs_seqs' in predictions file. Keys: {list(preds.keys())}")

    return seq_idx_map, preds


def _run_predict_evo2(
    *,
    fasta_path: Path,
    ckpt_dir: Path,
    output_dir: Path,
    cuda_visible_devices: str,
    model_size: str,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
):
    cmd = [
        "predict_evo2",
        "--fasta",
        str(fasta_path),
        "--ckpt-dir",
        str(ckpt_dir),
        "--output-dir",
        str(output_dir),
        "--model-size",
        model_size,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_model_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
        "--output-log-prob-seqs",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)