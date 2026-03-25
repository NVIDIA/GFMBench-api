# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset, Subset
import pandas as pd


class BaseGFMTask(ABC):

    
    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None) -> None:
        """
        General task initialization.
        
        Args:
            root_data_dir_path: path to root data directory (where all data directories are located)
            task_config: optional configuration dictionary with user's settings.
                         Supported keys:
                         - "max_sequence_length": int or None - maximum sequence length for the model.
                           If provided, sequences will be truncated to min(task_default, max_sequence_length).
                         - "batch_size": int - batch size for dataloaders (default: 32).
                         - "max_num_samples": int or None - maximum number of samples to load per dataset.
                           If provided, datasets will be limited to min(max_num_samples, original_size).
                           If None or not provided, all samples are loaded.
        """
        self.root_data_dir_path: str = root_data_dir_path
        self.task_config: Optional[Dict[str, Any]] = task_config or {}
        
        # Extract batch_size from task_config (default: 32)
        self.batch_size: int = self.task_config.get("batch_size", 32)
        
        # Extract max_num_samples from task_config (default: None = use all samples)
        self.max_num_samples: Optional[int] = self.task_config.get("max_num_samples", None)
        
        # Extract disable_safe_model_call from task_config (default: False)
        self.disable_safe_model_call: bool = self.task_config.get("disable_safe_model_call", False)
        
        # Compute effective max_sequence_length
        self.max_sequence_length: int = self._get_max_sequence_length()

        # Create datasets (not dataloaders)
        self.train_dataset, self.validation_dataset, self.test_dataset = self._create_datasets()
    
    def _get_max_sequence_length(self) -> int:
        """
        Compute the effective maximum sequence length.
        
        Takes the minimum of:
        - Task's default max length (from _get_default_max_seq_len())
        - User's config max_sequence_length (if provided)
        
        Returns:
            int: Effective maximum sequence length
        """
        task_default_max = self._get_default_max_seq_len()
        config_max = self.task_config.get("max_sequence_length", None)
        
        if config_max is not None:
            return min(task_default_max, config_max)
        return task_default_max
    
    @abstractmethod
    def _get_default_max_seq_len(self) -> int:
        """
        Subclasses must implement this: return the task's default maximum sequence length.
        
        This is the inherent maximum sequence length for the task's data.
        The base class will compute the effective max_sequence_length by taking
        the minimum of this value and the user's config (if provided).
        
        Returns:
            int: Task's default maximum sequence length in base pairs
        """
        pass
    
    
    def _safe_model_call(self, model: Any, method_name: str, *args: Any, num_outputs: int = 1) -> Tuple[Any, ...]:
        """
        Safely call a model method if it exists, otherwise return None for each output.
        
        This wrapper allows tasks to call model methods that may not be implemented
        by all models, handling the case where a method doesn't exist gracefully.
        
        Args:
            model: The model instance to call the method on
            method_name: str, name of the method to call
            *args: Positional arguments to pass to the method
            num_outputs: int, number of output values expected from the method
            
        Returns:
            tuple: The method's return values, or tuple of None values if method doesn't exist
        """
        if self.disable_safe_model_call:
            # Call method directly without any checks, allowing exceptions to propagate
            result = getattr(model, method_name)(*args)
            if num_outputs == 1:
                return (result,)
            return result
        
        # Safe mode: check if method exists and is callable
        if hasattr(model, method_name) and callable(getattr(model, method_name)):
            try:
                result = getattr(model, method_name)(*args)
                # If single output expected, wrap in tuple
                if num_outputs == 1:
                    return (result,)
                return result
            except Exception as e:
                # If method exists but fails, return None values
                return tuple([None] * num_outputs)
        else:
            # Method doesn't exist, return None for each expected output
            return tuple([None] * num_outputs)

    @abstractmethod
    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Subclasses must implement this: return the fine-tune dataset"""
        pass

    @abstractmethod
    def get_task_attributes(self) -> Dict[str, Any]:
        """Subclasses must implement this: return task attributes as a dictionary.
        
        Required keys:
            "has_finetuning_data": [bool] - whether this task has training data for fine-tuning
            "has_validation_data": [bool] - whether this task has validation data
            "is_variant_effect_prediction": [bool] - whether this is a variant effect prediction task
            "num_labels": [int] - number of labels for classification (only for classification tasks)
            "is_snv_only_variants": [bool] - whether this task uses only SNV variants (only for zero-shot variant tasks)
            "conditional_input_metadata": [Optional[pd.DataFrame]] - metadata schema for conditional inputs,
                or None if task has no metadata. Should call get_conditional_input_meta_data_frame().
        """
        pass

    @abstractmethod
    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """
        Subclasses must implement this: return metadata schema for conditional inputs.
        
        If the task has additional metadata inputs (e.g., tissue type, demographics),
        return a DataFrame describing them. If the task has no metadata, return None.
        
        The conditional input values will be passed to model inference methods as a tensor
        of shape (batch_size, num_metadata_inputs) where num_metadata_inputs equals and in the same order as the
        rows in this DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with the following columns:
                - 'meta_data_name': str - name of the metadata attribute (e.g., 'tissue_id', 'gender')
                - 'data_type': str - type of the data ('integer' or 'float')
                - 'min_value': numeric - minimum expected value for this attribute
                - 'max_value': numeric - maximum expected value for this attribute
            Each row represents one metadata attribute. Returns None if task has no metadata.
        """
        pass

    @abstractmethod
    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """Subclasses must implement this: evaluate the model on the given dataset and return scores.
        
        Args:
            model: Any object with inference methods. Uses _safe_model_call to handle missing methods gracefully.
            dataset: The dataset to evaluate on.
            
        Returns:
            dict: Scores in a dictionary: [metric_name: score]
        """
        pass

    def eval_test_set(self, model: Any) -> Dict[str, Optional[float]]:
        """Evaluate the model on the test dataset.
        
        Args:
            model: Any object with inference methods.
            
        Returns:
            dict: Test scores with metric names as keys.
        """
        return self._eval_dataset(model, self.test_dataset)

    def eval_validation_set(self, model: Any) -> Dict[str, Optional[float]]:
        """Evaluate the model on the validation dataset.
        
        Args:
            model: Any object with inference methods.
            
        Returns:
            dict: Validation scores with metric names as keys.
            
        Raises:
            ValueError: If validation dataset does not exist.
        """
        if self.validation_dataset is None:
            raise ValueError(
                f"Validation dataset does not exist for task '{self.get_task_name()}'. "
                "This task does not have a validation set."
            )
        return self._eval_dataset(model, self.validation_dataset)

    def eval_cross_validation_fold(self, model: Any, train_indices: List[int]) -> Dict[str, Optional[float]]:
        """Evaluate the model on a subset of the training dataset for cross-validation.
        
        Args:
            model: Any object with inference methods.
            train_indices: List of indices into the training dataset to use for evaluation.
            
        Returns:
            dict: Scores with metric names as keys.
            
        Raises:
            ValueError: If training dataset does not exist or indices are out of range.
        """
        if self.train_dataset is None:
            raise ValueError(
                f"Training dataset does not exist for task '{self.get_task_name()}'. "
                "Cross-validation requires a training set."
            )
        
        train_len = len(self.train_dataset)
        
        # Validate indices
        if not train_indices:
            raise ValueError("train_indices cannot be empty.")
        
        max_idx = max(train_indices)
        min_idx = min(train_indices)
        
        if max_idx >= train_len:
            raise ValueError(
                f"Index {max_idx} is out of range for training dataset of length {train_len}."
            )
        if min_idx < 0:
            raise ValueError(
                f"Index {min_idx} is negative. All indices must be non-negative."
            )
        
        # Create subset of training dataset
        subset = Subset(self.train_dataset, train_indices)
        
        return self._eval_dataset(model, subset)

    @abstractmethod
    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        """Subclasses must implement this: return train_dataset, validation_dataset, test_dataset"""
        pass

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Subclasses must implement this: return the task name.
        
        The task name should be identical to the data directory name containing
        the task's data files.
        
        Returns:
            str: Task name (e.g., 'gue_promoter_all', 'bend_vep_disease')
        """
        pass
