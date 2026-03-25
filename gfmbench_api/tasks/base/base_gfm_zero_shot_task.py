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
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import (
    SequenceEmbeddingsCosineSimAUPRC,
    SequenceEmbeddingsCosineSimAUROC,
    SequenceEmbeddingsL2AUPRC,
    SequenceEmbeddingsL2AUROC,
    SumProbsLLRAUPRC,
    SumProbsLLRAUROC
)
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class BaseGFMZeroShotTask(BaseGFMTask):
    """
    Base class for zero-shot variant effect prediction tasks.
    No fine-tuning is performed - evaluates model's pre-trained capabilities.
    
    Dataset format: (variant_sequence, reference_sequence, label, conditional_input) tuples
    
    Subclasses must implement:
        - _get_additional_metrics(): Return list of additional metric objects
        - _update_additional_metrics(): Update additional metrics for a batch
        - _is_snv_only(): Return whether this is an SNV-only task
        - _create_test_dataset(): Return test dataset
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema or None
    """
    
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize zero-shot variant task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """
        super().__init__(root_data_dir_path, task_config)

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return None (zero-shot tasks don't require fine-tuning)."""
        return None

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for zero-shot variant tasks."""
        return {
            "has_finetuning_data": False,
            "has_validation_data": False,
            "is_variant_effect_prediction": True,
            "is_snv_only_variants": self._is_snv_only(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame()
        }

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        """Create datasets for zero-shot task (test only, no training or validation)."""
        test_dataset = self._create_test_dataset()
        return None, None, test_dataset

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model using zero-shot methods for variant pathogenicity prediction.
        Binary classification task (benign=0, pathogenic=1).
        
        Common metrics (defined in base class):
        - LLR-based (using sequence probabilities): auroc_all_probs_llr, auprc_all_probs_llr
        - Cosine similarity on sequence embeddings: sequence_embeddings_cosinesim_auroc/auprc
        
        Additional metrics are provided by subclasses via _get_additional_metrics().
        
        Args:
            model: Any object with inference methods (should be in eval mode).
            dataset: The dataset to evaluate on.
        
        Returns:
            dict: Scores with metric names as keys.
        """
        # Create dataloader from dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Define common metrics with their argument keys
        # Each entry: (metric, arg_keys) where arg_keys specifies which outputs to pass
        common_metrics = [
            (SumProbsLLRAUROC(), ('variant_probs', 'reference_probs', 'labels')),
            (SumProbsLLRAUPRC(), ('variant_probs', 'reference_probs', 'labels')),
            (SequenceEmbeddingsCosineSimAUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsCosineSimAUPRC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUPRC(), ('variant_repr', 'reference_repr', 'labels')),
        ]
        
        # Get additional metrics from subclass
        additional_metrics = self._get_additional_metrics()

        for batch_data in tqdm(data_loader, desc="Evaluating (Zero-Shot)"):
            variant_sequences, reference_sequences, labels, conditional_input = batch_data
            
            # Get sequence-to-sequence outputs for both variant and reference
            # Returns: (sequence_probs, sequence_embeddings, sequence_representative)
            variant_probs_np, variant_embeddings_np, variant_repr_np = self._safe_model_call(
                model, 'infer_sequence_to_sequence', variant_sequences, conditional_input, num_outputs=3
            )
            reference_probs_np, reference_embeddings_np, reference_repr_np = self._safe_model_call(
                model, 'infer_sequence_to_sequence', reference_sequences, conditional_input, num_outputs=3
            )
            
            # Build outputs dict for metric argument lookup
            outputs = {
                'variant_probs': variant_probs_np,
                'reference_probs': reference_probs_np,
                'variant_embeddings': variant_embeddings_np,
                'reference_embeddings': reference_embeddings_np,
                'variant_repr': variant_repr_np,
                'reference_repr': reference_repr_np,
                'labels': labels,
            }
            
            # Update common metrics using their specified argument keys
            for metric, arg_keys in common_metrics:
                args = [outputs[key] for key in arg_keys]
                metric.calc(*args)
            
            # Update additional metrics (subclass handles this)
            self._update_additional_metrics(
                additional_metrics, model, 
                variant_sequences, reference_sequences, labels,
                outputs
            )

        # Aggregate results from all metrics
        results = {}
        for metric, _ in common_metrics:
            results[metric.name] = metric.get_final_results()
        for metric, _ in additional_metrics:
            results[metric.name] = metric.get_final_results()
        
        return results

    @abstractmethod
    def _get_additional_metrics(self) -> List[tuple]:
        """
        Subclasses must implement this: return list of additional metrics with arg_keys.
        
        Each entry is a tuple: (metric, arg_keys)
        arg_keys specifies which outputs to pass to metric.calc()
        
        Returns:
            List of (metric, arg_keys) tuples (can be empty)
        """
        pass

    @abstractmethod
    def _update_additional_metrics(
        self,
        additional_metrics: List[Any],
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        labels_np: np.ndarray,
        common_outputs: Dict[str, Optional[np.ndarray]]
    ) -> None:
        """
        Subclasses must implement this: update additional metrics for a batch.
        
        This method is called once per batch. Subclasses can make additional
        model calls if needed (e.g., for SNV-specific metrics).
        
        Args:
            additional_metrics: List of metric objects from _get_additional_metrics()
            model: Model instance for additional inference calls
            variant_sequences: Batch of variant sequences
            reference_sequences: Batch of reference sequences
            labels_np: Labels as numpy array
            common_outputs: Dict with common model outputs (probs, embeddings, repr)
        """
        pass

    @abstractmethod
    def _is_snv_only(self) -> bool:
        """Subclasses must implement this: return whether this is an SNV-only task."""
        pass

    @abstractmethod
    def _create_test_dataset(self) -> Dataset:
        """
        Subclasses must implement this: create and return test dataset.
        The dataset must yield (variant_sequences, reference_sequences, labels, conditional_input) tuples.
        conditional_input should be None for tasks without metadata.
        
        Returns:
            Dataset: test dataset with (variant_sequences, reference_sequences, labels, conditional_input) tuples
        """
        pass

