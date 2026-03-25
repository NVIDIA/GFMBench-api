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
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import glob
import gzip
import logging
import os
import pandas as pd
import shutil
from typing import Callable, Iterable, List, Optional, Literal, Union

from datasets import load_dataset, get_dataset_config_names, concatenate_datasets, DatasetDict

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

# =============================================================================
# HuggingFace Dataset Utilities
# =============================================================================

def download_hf_dataset_files(
    repo_id: str,
    subfolder: str | list[str],
    local_dir: str,
    splits: list[str] = ["train", "test", "dev"],
    concat_tasks: bool = False,
    repo_type: str = "dataset",
) -> None:
    """Download specified Hugging Face dataset subset if not already present, saving as CSV or Parquet."""
    os.makedirs(local_dir, exist_ok=True)

    try:
        DatasetDict.load_from_disk(local_dir)
        return # if dataset is already downloaded to the local dir
    except:
        logging.info('Dataset not found locally, downloading')

    config_names = get_dataset_config_names(repo_id)

    if concat_tasks:
        if not isinstance(subfolder, list):
            raise TypeError("When concat_tasks is True, subset_name must be a list of subset names.")

        all_splits = {split: [] for split in splits}
        for subset in subfolder:
            if subset not in config_names:
                raise ValueError(f"Expected subset {subset} not found in HF dataset configs.")
            ds_dict = load_dataset(repo_id, subset)
            for split in splits:
                if split in ds_dict:
                    all_splits[split].append(ds_dict[split])

        for split, pieces in all_splits.items():
            if not pieces:
                continue
            combined = concatenate_datasets(pieces)
            df = combined.to_pandas()
            split_path = os.path.join(local_dir, f"{split}.csv")
            df.to_csv(split_path, index=False)

    else:
        if not isinstance(subfolder, str):
            raise TypeError("When concat_tasks is False, subset_name must be a single subset name.")
        if subfolder not in config_names:
            raise ValueError(f"Expected subset {subfolder} not found in HF dataset configs.")

        ds_dict = load_dataset(repo_id, subfolder)
        ds_dict.save_to_disk(local_dir)

def iter_subset_dataframes(
    task_path: str,
    *,
    exts: Iterable[str] = ("csv", "parquet"),
    columns: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    """
    Yield DataFrames for each subset file under `task_path` with supported extensions.

    - Reads only `columns` when possible.
    - Searches recursively for parquet to support HF-style dataset repos
      (often parquet lives in nested folders). CSV is kept to top-level to
      preserve original behavior, but you can make it recursive too if you want.
    """
    if not os.path.isdir(task_path):
        raise ValueError(f"task_path must be a directory, got: {task_path}")

    files: List[str] = []
    if "csv" in exts:
        files.extend(glob.glob(os.path.join(task_path, "*.csv")))
    if "parquet" in exts:
        files.extend(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            # usecols raises ValueError if missing -> let caller decide; here we just load
            df = pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
        else:
            continue
        yield df

def reduce_over_subsets(task_path: str, columns: List[str], reducer: Callable[[pd.DataFrame], int]) -> int:
    """Shared driver to avoid duplicating subset iteration and file reading."""
    best = 0
    for df in iter_subset_dataframes(task_path, columns=columns):
        val = reducer(df)
        if val > best:
            best = val
    return best


def get_max_sequence_length_for_task(task_path: str, sequence_col_name: str = "sequence") -> int:
    """Return maximum character length of `sequence_col_name` across all subsets (*.csv, *.parquet)."""
    def _subset_max(df: pd.DataFrame) -> int:
        if sequence_col_name not in df.columns or df.empty:
            return 0
        s = df[sequence_col_name].astype("string", copy=False)
        m = s.str.len().max()
        return int(m) if pd.notna(m) else 0

    return reduce_over_subsets(task_path, columns=[sequence_col_name], reducer=_subset_max)


def get_num_classes_for_task(task_path: str, label_col_name: str = "label") -> int:
    """Return number of unique labels in `label_col_name` across all subsets (*.csv, *.parquet)."""
    unique_labels = set()

    for df in iter_subset_dataframes(task_path, columns=[label_col_name]):
        if label_col_name not in df.columns or df.empty:
            continue
        unique_labels.update(df[label_col_name].dropna().unique().tolist())

    return len(unique_labels)

# =============================================================================
# Generic URL Downloads
# =============================================================================

def download_file_from_url(url: str, local_path: str) -> None:
    """
    Download a file from a URL with progress reporting.
    
    Args:
        url: URL to download from
        local_path: Local path to save the file to
    """
    import urllib.request
    
    # Create parent directory if needed
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    
    logging.info(f"Downloading from: {url}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}% ({downloaded / 1e6:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, local_path, reporthook)
    print()  # New line after progress (keep as print for carriage return handling)

# =============================================================================
# Reference Genome Download
# =============================================================================

UCSC_HG38_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"


def ensure_reference_genome(reference_genome_path: str) -> None:
    """
    Download hg38 reference genome if it doesn't exist.
    
    Downloads from UCSC (~3GB compressed, ~3.1GB uncompressed).
    
    Args:
        reference_genome_path: Path where hg38.fa should be saved
    """
    if os.path.exists(reference_genome_path):
        return
    
    ref_dir = os.path.dirname(reference_genome_path)
    os.makedirs(ref_dir, exist_ok=True)
    
    gz_path = reference_genome_path + ".gz"
    
    logging.info(f"Downloading reference genome hg38.fa (~3GB)...")
    logging.info(f"Source: {UCSC_HG38_URL}")
    logging.info("This may take several minutes...")
    
    download_file_from_url(UCSC_HG38_URL, gz_path)
    
    logging.info("Extracting hg38.fa.gz...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(reference_genome_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(gz_path)
    logging.info(f"Reference genome saved to: {reference_genome_path}")

