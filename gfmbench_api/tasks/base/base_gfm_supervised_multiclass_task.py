# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import (
    MultiLabelClassificationAccuracy,
    MultiLabelClassificationAUPRC,
    MultiLabelClassificationAUROC,
    MultiLabelClassificationMCC,
)

from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask


class BaseGFMSupervisedMultiClassTask(BaseGFMTask):
    """
    Base class for supervised multi-class classification tasks.
    Implements testing for standard classification tasks.
    
    Subclasses must implement:
        - _batch_to_probs(batch, model): Extract probs and labels from a batch
        - _get_num_labels(): Return number of classification labels
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
    """
    
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize multi-label task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """
        # Call parent initialization
        super().__init__(root_data_dir_path, task_config)

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return the training dataset for fine-tuning."""
        return self.train_dataset

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for classification tasks."""
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "is_variant_effect_prediction": self._is_variant_effect_prediction(),
            "num_labels": self._get_num_labels(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame()
        }
    
    @abstractmethod
    def _is_variant_effect_prediction(self) -> bool:
        """Subclasses must implement this: return whether this is a variant effect prediction task."""
        pass

    @abstractmethod
    def _batch_to_probs(
        self, batch: Any, model: Any
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract probabilities and labels from a batch using the model.
        
        Subclasses must implement this to handle their specific batch format:
        - Single sequence tasks: batch = (sequences, labels)
        - Variant effect tasks: batch = (ref_sequences, variant_sequences, labels)
        
        Args:
            batch: A batch from the DataLoader (tuple of tensors/lists)
            model: Model instance to use for inference
        
        Returns:
            Tuple of (probs, labels_np):
                - probs: np.ndarray of shape [batch_size, num_labels] or None
                - labels_np: np.ndarray of shape [batch_size]
        """
        pass

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on the given dataset.
        
        Args:
            model: Model instance to evaluate (must implement the appropriate inference method)
            dataset: The dataset to evaluate on.
        
        Returns:
            dict: Scores with metric names as keys:
                - 'classification_accuracy': Accuracy score (0-1)
                - 'classification_mcc': Matthews Correlation Coefficient
                - 'classification_auroc': Area Under ROC Curve
                - 'classification_auprc': Area Under Precision-Recall Curve
        """
        
        # Create dataloader from dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize metric classes
        metrics = [
            MultiLabelClassificationAccuracy(),
            MultiLabelClassificationMCC(),
            MultiLabelClassificationAUROC(),
            MultiLabelClassificationAUPRC()
        ]

        for batch in tqdm(data_loader, desc="Evaluating"):
            # Delegate to subclass for batch processing
            probs, labels_np = self._batch_to_probs(batch, model)
            
            # Verify model output is valid
            if probs is not None:
                # Verify that the number of classes matches num_labels
                num_labels = self._get_num_labels()
                assert probs.shape[1] == num_labels, \
                    f"Expected {num_labels} classes, but got {probs.shape[1]}"
                
                # Verify that probabilities sum to 1 (with epsilon tolerance)
                prob_sums = probs.sum(axis=1)
                epsilon = 1e-5
                assert np.allclose(prob_sums, np.ones_like(prob_sums), atol=epsilon), \
                    f"Probabilities do not sum to 1. Got sums in range [{prob_sums.min():.6f}, {prob_sums.max():.6f}]"
            
            # Calculate intermediate values for each metric
            for metric in metrics:
                metric.calc(probs, labels_np)

        # Get final results dynamically using metric.name as the result key
        results = {}
        for metric in metrics:
            score = metric.get_final_results()
            results[metric.name] = score
        
        return results
    
    @abstractmethod
    def _get_num_labels(self) -> int:
        """Subclasses must implement this: return number of classification labels"""
        pass
