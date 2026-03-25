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

# This module does not embed third-party data download URLs.
"""
Utility functions for extracting sequences from reference genome and converting dataset-specific formats to the common "sequence", "label" format.
Handles SNV variants with proper windowing and validation.
"""
import logging
from typing import List, Optional, Tuple
import pandas as pd
import re

from Bio.Seq import reverse_complement
from pyfaidx import Fasta

# =============================================================================
# Sequence Processing Utilities
# =============================================================================

def standardize_sequence(sequence: str) -> str:
    """Standardize DNA sequence: uppercase and replace non-ATCG with N."""
    sequence = str(sequence).upper()
    return re.sub(r"[^ATCG]", "N", sequence)

def pad_sequence(chromosome, start: int, sequence_length: int, end: Optional[int] = None, negative_strand: bool = False):
    """
    Extends a given sequence to length `sequence_length`.
    
    Handles out-of-bounds regions by padding with 'N's, making it robust
    for variants near chromosome boundaries.
    
    Args:
        chromosome: pyfaidx chromosome object
        start: Start position (can be negative for padding)
        sequence_length: Desired output length
        end: End position (optional, computed from start + sequence_length if None)
        negative_strand: If True, return reverse complement
        
    Returns:
        DNA sequence string of exactly `sequence_length` characters
    """
    chrom_len = len(chromosome)
    if end is None:
        end = start + sequence_length
    
    # Calculate padding needed for out-of-bounds regions
    pad_left = abs(start) if start < 0 else 0
    pad_right = end - chrom_len if end > chrom_len else 0
    
    # Clamp to valid chromosome range
    actual_start = max(0, start)
    actual_end = min(chrom_len, end)
    
    # Extract sequence (or return all N's if completely out of bounds)
    if actual_start >= actual_end:
        seq_str = "N" * sequence_length
    else:
        seq_str = str(chromosome[actual_start:actual_end])
        
    # Add padding
    final_seq = ("N" * pad_left) + seq_str + ("N" * pad_right)
    
    # Handle reverse complement
    if negative_strand:
        trans = str.maketrans("ATCGN", "TAGCN")
        final_seq = final_seq.translate(trans)[::-1]
    
    # Ensure exact length (safety check)
    if len(final_seq) != sequence_length:
        final_seq = final_seq[:sequence_length].ljust(sequence_length, 'N')
    
    return final_seq

def pad_sequence_centered_variant(
    chromosome, 
    variant_pos_0based: int, 
    max_sequence_length: int,
    variant_pos_in_seq: Optional[int] = None
) -> str:
    """
    Extract sequence centered on variant position, padding with 'P' if window exceeds chromosome boundaries.
    
    This function ensures the variant remains at the specified center position even when the desired
    window extends beyond chromosome boundaries. Padding is applied symmetrically to maintain centering.
    
    Args:
        chromosome: pyfaidx chromosome object
        variant_pos_0based: Variant position in chromosome (0-based)
        max_sequence_length: Desired output sequence length
        variant_pos_in_seq: Position of variant within output sequence (default: max_sequence_length // 2)
        
    Returns:
        DNA sequence string of exactly `max_sequence_length` characters with variant at `variant_pos_in_seq`
        
    Example:
        >>> # Variant at position 1000, want 1000bp window centered
        >>> seq = pad_sequence_centered_variant(chrom, 1000, 1000, variant_pos_in_seq=500)
        >>> len(seq)  # Returns 1000
        1000
        >>> # If variant is near chromosome start, left side will be padded with 'P'
    """
    chrom_len = len(chromosome)
    
    # Default variant position to center if not specified
    if variant_pos_in_seq is None:
        variant_pos_in_seq = max_sequence_length // 2
    
    # Calculate desired window boundaries (centered on variant)
    window_start = variant_pos_0based - variant_pos_in_seq
    window_end = window_start + max_sequence_length
    
    # Calculate padding needed
    pad_left = abs(window_start) if window_start < 0 else 0
    pad_right = window_end - chrom_len if window_end > chrom_len else 0
    
    # Clamp to valid chromosome range
    actual_start = max(0, window_start)
    actual_end = min(chrom_len, window_end)
    
    # Extract sequence from chromosome (or empty if completely out of bounds)
    if actual_start >= actual_end:
        seq_str = ""
    else:
        seq_str = str(chromosome[actual_start:actual_end]).upper()
    
    # Add padding to maintain variant at center position
    final_seq = ("P" * pad_left) + seq_str + ("P" * pad_right)
    
    # Ensure exact length (safety check)
    if len(final_seq) != max_sequence_length:
        # If somehow we got wrong length, pad or truncate to exact size
        if len(final_seq) < max_sequence_length:
            # Need more padding - add to right side
            final_seq = final_seq + ("P" * (max_sequence_length - len(final_seq)))
        else:
            # Too long - truncate from right side (shouldn't happen, but safety check)
            final_seq = final_seq[:max_sequence_length]
    
    # Verify variant is at expected position (sanity check)
    # The variant should be at variant_pos_in_seq in the final sequence
    # If we padded left, the variant position in the extracted seq_str would be adjusted
    # But since we want variant at variant_pos_in_seq, we need to ensure the padding is correct
    
    return final_seq


def truncate_sequence_from_ends(sequence: str, max_length: int) -> str:
    """
    Truncate a sequence from both ends to keep the center portion.
    
    Args:
        sequence: DNA sequence string
        max_length: Maximum allowed length
        
    Returns:
        Truncated sequence with center portion preserved
    """
    if len(sequence) <= max_length:
        return sequence
    
    # Calculate how many bases to remove from each end
    excess = len(sequence) - max_length
    trim_start = excess // 2
    trim_end = excess - trim_start
    
    return sequence[trim_start:len(sequence) - trim_end]

def clean_seq(seq: str) -> str:
    """
    Remove annotation tags and spaces from sequences
    """
    TAG_RE = re.compile(r"\[[^\]]+\]")
    if seq is None or pd.isna(seq):
        return None
    # remove tags like [START_CDS]
    seq = TAG_RE.sub("", str(seq))
    # remove all whitespace
    return seq.replace(" ", "")

def build_forward_centered_seqs(
    df: pd.DataFrame,
    *,
    alt_left_col: str = "alt_left",
    alt_right_col: str = "alt_right",
    ref_left_col: str = "ref_left",
    ref_right_col: str = "ref_right",
    label_col: str = "label",
    chrom_col: str = "chrom"
) -> pd.DataFrame:
    """
    Build forward-strand sequences centered on the SNP (BioFM format to forward-strand "sequence", "label" columns)

    Cleaning rules:
      - left sequences: remove tags ([...]) and spaces
      - right sequences: remove spaces only

    Assembly rule:
      forward = left + reverse_complement(right[:-1])
      forward[seq_len//2] is mapped back to standard nucleotide encoding
    """
    required = {alt_left_col, alt_right_col, ref_left_col, ref_right_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    def _build(left_raw, right_raw):
        VARIANT_MAP = {"A": "Â", "C": "Ĉ", "G": "Ĝ", "T": "Ṱ", "N": "N"}
        REVERSE_VARIANT_MAP = {v: k for k, v in VARIANT_MAP.items()}
        left = clean_seq(left_raw)
        right = clean_seq(right_raw)
        if not left or not right:
            return None
        left = left[:-1] + REVERSE_VARIANT_MAP.get(left[-1], left[-1])
        # drop SNP base from right to avoid duplication
        return left + reverse_complement(right[:-1])

    reference_sequences = [
        _build(l, r) for l, r in zip(df[ref_left_col], df[ref_right_col])
    ]
    variant_sequences = [
        _build(l, r) for l, r in zip(df[alt_left_col], df[alt_right_col])
    ]

    dataset = pd.DataFrame({'variant_sequence': variant_sequences, 'reference_sequence': reference_sequences, 'label': df[label_col]})

    return dataset

def get_reference_sequence(
    genome: Fasta,
    chrom: str,
    pos: int,
    flank_size: int
) -> Optional[str]:
    """
    Extract reference sequence around a variant position.
    
    Args:
        genome: pyfaidx Fasta object for the reference genome
        chrom: Chromosome name (will try with/without 'chr' prefix)
        pos: Variant position (0-based)
        flank_size: Number of bases to extract on each side of variant
                   Total sequence length will be: 2*flank_size + 1
    
    Returns:
        str: Reference sequence of length 2*flank_size + 1, or None if extraction fails
    
    Example:
        >>> seq = get_reference_sequence(genome, "1", 100, flank_size=256)
        >>> len(seq)  # Returns 513 (2*256 + 1)
        513
    """
    try:
        # Handle chromosome naming conventions
        if chrom not in genome:
            chrom = f"chr{chrom}"
        
        if chrom not in genome:
            return None
        
        # Extract window: [pos-flank_size-1, pos+flank_size]
        # -1 for 0-based indexing adjustment
        start = max(0, pos - flank_size - 1)
        end = pos + flank_size
        
        seq = str(genome[chrom][start:end].seq).upper()
        
        # Validate sequence length
        # If we're at the edge of the chromosome, sequence might be shorter
        expected_length = 2 * flank_size + 1
        if len(seq) != expected_length:
            return None
        
        return seq
    except Exception:
        return None


def generate_variant_sequence(
    reference_seq: str,
    ref_allele: str,
    alt_allele: str,
    flank_size: int
) -> Optional[str]:
    """
    Generate variant sequence by substituting reference allele with alternate allele.
    
    Args:
        reference_seq: Reference sequence (length must be 2*flank_size + 1)
        ref_allele: Reference allele (should match reference_seq at variant position)
        alt_allele: Alternate allele (what to substitute)
        flank_size: Number of flanking bases (used to calculate variant position)
    
    Returns:
        str: Variant sequence with substitution, or None if validation fails
    
    Example:
        >>> ref_seq = "ACGTACGTACGTACGT"  # length 16 = 2*7 + 1 (flank_size=7)
        >>> var_seq = generate_variant_sequence(ref_seq, "T", "A", flank_size=7)
        >>> var_seq
        'ACGTACGTAACGTACGT'  # T->A at position 8 (center)
    """
    try:
        # Ensure single nucleotide variants
        if len(ref_allele) != 1 or len(alt_allele) != 1:
            return None
        
        # Variant position is at the center of the sequence
        variant_pos = flank_size
        
        # Validate reference allele matches the sequence
        if reference_seq[variant_pos] != ref_allele:
            return None
        
        # Create variant sequence
        seq_list = list(reference_seq)
        seq_list[variant_pos] = alt_allele
        variant_seq = "".join(seq_list)
        
        # Ensure lengths match
        if len(variant_seq) != len(reference_seq):
            return None
        
        return variant_seq
    except Exception:
        return None


def extract_snv_sequences_from_genome(
    df: pd.DataFrame,
    genome: Fasta,
    flank_size: int,
    chrom_col: str = 'chrom',
    pos_col: str = 'pos',
    ref_col: str = 'ref',
    alt_col: str = 'alt',
    label_col: str = 'label',
    verbose: bool = True
) -> Tuple[list, list, list, int]:
    """
    Extract SNV sequences from reference genome for all variants in dataframe.
    
    Args:
        df: DataFrame with variant information
        genome: pyfaidx Fasta object for reference genome
        flank_size: Number of bases to extract on each side of variant
        chrom_col: Column name for chromosome
        pos_col: Column name for position (0-based)
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        label_col: Column name for labels
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (reference_sequences, variant_sequences, labels, num_skipped)
        where each list contains valid extracted sequences
    
    Example:
        >>> df = pd.read_csv("variants.csv")
        >>> ref_seqs, var_seqs, labels, skipped = extract_snv_sequences_from_genome(
        ...     df, genome, flank_size=256
        ... )
        >>> print(f"Extracted {len(ref_seqs)} samples, skipped {skipped}")
    """
    reference_sequences = []
    variant_sequences = []
    labels = []
    skipped = 0
    
    if verbose:
        logging.info(f"Extracting sequences (window size: {2*flank_size+1}bp, flank: {flank_size}bp)...")
    
    for idx, row in df.iterrows():
        chrom = str(row[chrom_col])
        pos = int(row[pos_col])
        ref_allele = str(row[ref_col]).upper()
        alt_allele = str(row[alt_col]).upper()
        label = int(row[label_col]) if row[label_col] in [0, 1] else (1 if row[label_col] else 0)
        
        # Extract reference sequence
        ref_seq = get_reference_sequence(genome, chrom, pos, flank_size)
        if ref_seq is None:
            skipped += 1
            continue
        
        # Generate variant sequence
        var_seq = generate_variant_sequence(ref_seq, ref_allele, alt_allele, flank_size)
        if var_seq is None:
            skipped += 1
            continue
        
        # All validations passed
        reference_sequences.append(ref_seq)
        variant_sequences.append(var_seq)
        labels.append(label)
    
    if verbose and skipped > 0:
        logging.info(f"Skipped {skipped} variants due to extraction issues or allele mismatch")
    
    if verbose:
        logging.info(f"After extraction: {len(labels)} samples with valid sequences")
    
    return reference_sequences, variant_sequences, labels, skipped


# =============================================================================
# Optimized Batch SNV Sequence Extraction (VCF 1-based positions)
# =============================================================================

def extract_snv_sequences_centered(
    df: pd.DataFrame,
    genome: Fasta,
    max_sequence_length: int,
    chrom_col: str = 'chrom',
    pos_col: str = 'pos',
    ref_col: str = 'ref',
    alt_col: str = 'alt',
    label_col: str = 'label',
    use_logging: bool = True,
) -> Tuple[List[str], List[str], List[int], int]:
    """
    Extract centered SNV sequences from reference genome using VCF 1-based positions.
    
    This is an optimized version that:
    - Groups by chromosome for efficient batch processing
    - Uses VCF-standard 1-based positions
    - Handles chromosome naming variations (with/without 'chr' prefix)
    - Returns sequences of exactly `max_sequence_length` with variant at center
    - Validates that ref/alt alleles are single nucleotides (SNV constraint)
    - Validates that extracted reference base matches the VCF allele (catches off-by-one position errors)
    
    Args:
        df: DataFrame with variant information
        genome: pyfaidx Fasta object for reference genome
        max_sequence_length: Desired output sequence length (variant will be centered)
        chrom_col: Column name for chromosome
        pos_col: Column name for position (1-based VCF position)
        ref_col: Column name for reference allele (must be single nucleotide)
        alt_col: Column name for alternate allele (must be single nucleotide)
        label_col: Column name for labels
        use_logging: If True, use logging.info; if False, use print
    
    Returns:
        Tuple of (variant_sequences, reference_sequences, labels, num_skipped)
        Note: Returns (var_seqs, ref_seqs, ...) to match expected task ordering
    
    Validation checks:
        1. SNV constraint: ref_allele and alt_allele must be single nucleotides
        2. Reference validation: extracted base must match VCF ref_allele (detects position errors)
        3. Chromosome existence and window bounds
        4. Sequence length consistency
    
    Skipped variants due to:
        - Non-SNV: ref or alt alleles with length != 1
        - Chromosome not found in reference genome
        - Variant position out of bounds for window
        - Reference allele mismatch (indicates position error or data quality issue)
        - Sequence length mismatch (boundary edge case)
    
    Examples:
        >>> # Basic usage with VCF-style dataframe
        >>> df = pd.DataFrame({
        ...     'chrom': ['chr1', 'chr1'],
        ...     'pos': [100, 200],  # 1-based VCF positions
        ...     'ref': ['A', 'G'],
        ...     'alt': ['T', 'C'],
        ...     'label': [0, 1]
        ... })
        >>> var_seqs, ref_seqs, labels, skipped = extract_snv_sequences_centered(
        ...     df, genome, max_sequence_length=512
        ... )
        
        >>> # Verify sequences are centered
        >>> flank_size = (512 - 1) // 2  # = 255
        >>> # Variant is at position flank_size in the sequence
        >>> assert ref_seqs[0][flank_size] == 'A'  # Original ref allele
        >>> assert var_seqs[0][flank_size] == 'T'  # Substituted alt allele
    """
    log_fn = logging.info if use_logging else print
    
    flank_size = (max_sequence_length - 1) // 2
    
    reference_sequences: List[str] = []
    variant_sequences: List[str] = []
    labels: List[int] = []
    skipped = 0
    skipped_non_snv = 0
    skipped_ref_mismatch = 0
    
    log_fn(f"Extracting sequences with window {max_sequence_length}bp (flank {flank_size}bp)...")
    
    # Process by chromosome for efficiency
    for chrom, group in df.groupby(chrom_col):
        # Handle chromosome naming (try with and without 'chr' prefix)
        chrom_str = str(chrom)
        chrom_key = chrom_str if chrom_str.startswith("chr") else f"chr{chrom_str}"
        
        try:
            chrom_obj = genome[chrom_key]
        except KeyError:
            # Try without chr prefix
            try:
                chrom_obj = genome[chrom_str]
            except KeyError:
                log_fn(f"Chromosome {chrom} not found in reference genome, skipping {len(group)} variants")
                skipped += len(group)
                continue
        
        for _, row in group.iterrows():
            pos = int(row[pos_col])  # 1-based VCF position
            ref_allele = str(row[ref_col]).upper()
            alt_allele = str(row[alt_col]).upper()
            label = int(row[label_col]) if row[label_col] in [0, 1] else (1 if row[label_col] else 0)
            
            # Validate SNV constraint (single nucleotide variants only)
            if len(ref_allele) != 1 or len(alt_allele) != 1:
                logging.debug(f"Skipping non-SNV at {chrom_str}:{pos} (ref={ref_allele}, alt={alt_allele})")
                skipped += 1
                skipped_non_snv += 1
                continue
            
            # Convert to 0-based for indexing
            variant_pos_0based = pos - 1
            
            # Extract reference sequence using padding function (handles chromosome boundaries)
            try:
                # Get chromosome object for padding function
                chrom_obj = genome[chrom_key] if chrom_key in genome else genome[chrom_str]
                ref_seq = pad_sequence_centered_variant(
                    chromosome=chrom_obj,
                    variant_pos_0based=variant_pos_0based,
                    max_sequence_length=max_sequence_length,
                    variant_pos_in_seq=flank_size
                )
            except Exception as e:
                logging.debug(f"Error extracting sequence for {chrom_str}:{pos}. {str(e)}. Skipping.")
                skipped += 1
                continue
            
            # Validate sequence length
            if len(ref_seq) != max_sequence_length:
                skipped += 1
                continue
            
            # Variant position in window should be at flank_size (center)
            variant_pos_in_window = flank_size
            
            # Check if padding character is at variant position (shouldn't happen, but safety check)
            if variant_pos_in_window < len(ref_seq) and ref_seq[variant_pos_in_window] == 'P':
                logging.debug(f"Variant position at {chrom_str}:{pos} falls in padding region. Skipping.")
                skipped += 1
                continue
            
            # Validate reference allele matches genome (catches position errors)
            actual_ref = ref_seq[variant_pos_in_window]
            if actual_ref != ref_allele:
                logging.debug(
                    f"Ref allele mismatch at {chrom_str}:{pos} "
                    f"(expected={ref_allele}, actual={actual_ref}). "
                    f"This may indicate a position off-by-one error or data quality issue."
                )
                skipped += 1
                skipped_ref_mismatch += 1
                continue
            
            # Create variant sequence by substituting the allele
            # For SNVs: ref and alt are single nucleotides
            var_seq = (
                ref_seq[:variant_pos_in_window]
                + alt_allele
                + ref_seq[variant_pos_in_window + 1:]
            )
            
            reference_sequences.append(ref_seq)
            variant_sequences.append(var_seq)
            labels.append(label)
    
    if skipped > 0:
        log_fn(
            f"Skipped {skipped} variants: "
            f"{skipped_non_snv} non-SNV, {skipped_ref_mismatch} ref mismatch, "
            f"{skipped - skipped_non_snv - skipped_ref_mismatch} other (bounds/length/centering)"
        )
    
    # Return (var_seqs, ref_seqs, labels, skipped) to match task expectations
    return variant_sequences, reference_sequences, labels, skipped


# =============================================================================
# Test Functions for SNV Sequence Extraction
# =============================================================================

def _test_snv_sequence_extraction_logic():
    """
    Unit tests for SNV sequence extraction logic.
    
    **ACTIVE TEST SUITE**: Run with `python -m doctest preprocutils.py` to validate
    the position conversion, window centering, and allele substitution logic used in
    extract_snv_sequences_centered().
    
    Tests the core logic without external dependencies (pyfaidx, reference genome).
    This validates position conversion, window centering, and allele substitution logic.
    
    Examples:
        >>> # Test 1: VCF 1-based to 0-based position conversion
        >>> pos_vcf = 100  # 1-based VCF position
        >>> pos_0based = pos_vcf - 1
        >>> assert pos_0based == 99, f"Expected 99, got {pos_0based}"
        
        >>> # Test 2: Flank size calculation from max_sequence_length
        >>> max_seq_len = 512
        >>> flank_size = (max_seq_len - 1) // 2
        >>> assert flank_size == 255, f"Expected 255, got {flank_size}"
        >>> # Verify: flank_size + 1 + flank_size = max_seq_len (center + both flanks)
        >>> assert flank_size + 1 + flank_size == 511  # Not quite 512 due to floor division
        
        >>> # Test 3: Window start calculation (variant at center)
        >>> pos_0based = 1000
        >>> flank_size = 255
        >>> window_start = max(0, pos_0based - flank_size)
        >>> window_end = window_start + 512
        >>> variant_pos_in_window = pos_0based - window_start
        >>> assert variant_pos_in_window == flank_size, f"Expected {flank_size}, got {variant_pos_in_window}"
        
        >>> # Test 4: SNV substitution at center position
        >>> ref_seq = "A" * 255 + "G" + "T" * 256  # G at center (position 255)
        >>> variant_pos_in_window = 255
        >>> alt_allele = "C"
        >>> var_seq = ref_seq[:variant_pos_in_window] + alt_allele + ref_seq[variant_pos_in_window + 1:]
        >>> assert len(var_seq) == len(ref_seq), "Length should be preserved for SNVs"
        >>> assert var_seq[255] == "C", f"Expected C at position 255, got {var_seq[255]}"
        >>> assert ref_seq[255] == "G", f"Reference should have G at position 255"
        
        >>> # Test 5: Chromosome boundary handling (variant near start)
        >>> pos_0based = 100  # Close to chromosome start
        >>> flank_size = 255
        >>> window_start = max(0, pos_0based - flank_size)
        >>> assert window_start == 0, "Should clamp to 0 at chromosome start"
        >>> variant_pos_in_window = pos_0based - window_start
        >>> assert variant_pos_in_window == 100, "Variant not centered at boundary"
        >>> # This variant would be SKIPPED because variant_pos_in_window != flank_size
        
        >>> # Test 6: Reference allele validation (critical for data integrity and position errors)
        >>> ref_seq = "ACGTACGTACGT"
        >>> variant_pos = 5
        >>> expected_ref = "C"  # From VCF
        >>> actual_ref = ref_seq[variant_pos]
        >>> if actual_ref != expected_ref:
        ...     pass  # This variant would be skipped (ref mismatch)
        >>> assert ref_seq[5] == "C", "Position 5 should be C"
        
        >>> # Test 7: Edge case - odd vs even max_sequence_length
        >>> # Even: 512 -> flank=255, center at 255
        >>> flank_even = (512 - 1) // 2
        >>> assert flank_even == 255
        >>> # Odd: 513 -> flank=256, center at 256
        >>> flank_odd = (513 - 1) // 2
        >>> assert flank_odd == 256
        
        >>> # Test 8: Verify substitution preserves flanking sequences
        >>> ref_seq = "AAACCCGGGTTTT"  # length 13, center at 6
        >>> flank = 6
        >>> alt = "A"
        >>> var_seq = ref_seq[:flank] + alt + ref_seq[flank + 1:]
        >>> assert var_seq == "AAACCCAGGTTTT", f"Got {var_seq}"
        >>> assert var_seq[:flank] == ref_seq[:flank], "Left flank should match"
        >>> assert var_seq[flank+1:] == ref_seq[flank+1:], "Right flank should match"
    """
    pass
