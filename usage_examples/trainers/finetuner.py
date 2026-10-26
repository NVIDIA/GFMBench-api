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
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from gfmbench_api.utils.binning import get_valid_token_counts, pool_tokens_to_bins, to_float_tensor
from gfmbench_api.utils.caching_utils import SequenceInferenceCache
from usage_examples.trainers.model_wrapper import GFMWithProjection


class GFMFinetuner:
    """
    Fine-tuning module for genomic foundation models.
    Handles training of projection layers and full models for classification tasks.
    """
    
    def __init__(
        self, 
        model, 
        train_loader,
        hidden_dim,
        num_outputs,
        num_epochs,
        lr,
        optimizer_name="AdamW",
        weight_decay=0.01,
        only_proj_layer=True,
        classification_mode: str = "single_label",
        num_labels: int = 1,
        num_classes: int = 2,
        is_variant_effect_prediction=False,
        cache_size: Optional[float] = None,
        task_type: str = "classification",
        num_bins: int = 1,
        device='cpu'
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model: BaseGFMModel instance
            train_loader: DataLoader for training data
            hidden_dim: hidden dimension size of the model
            num_outputs: projection-head width
            num_epochs: number of training epochs
            lr: learning rate
            optimizer_name: optimizer name ('Adam', 'AdamW', 'SGD')
            weight_decay: weight decay for regularization
            only_proj_layer: if True, only train projection layer; if False, train full model
            classification_mode: classification target semantics
            num_labels: number of independent classification targets
            num_classes: number of classes per target
            is_variant_effect_prediction: if True, task uses variant/ref sequence pairs
            cache_size: cache RAM limit in GB; None for unlimited, 0 disables caching
            task_type: "classification" or "regression". Regression trains a linear head
                with MSE loss on the frozen backbone's public infer_sequence_to_sequence
                outputs; it requires only_proj_layer=True. For binned regression
                (num_bins > 1) the head is applied per token and the outputs are pooled
                into bins with the same helper the task uses at evaluation.
            num_bins: number of output bins of a regression task (1 = sequence-level)
            device: torch device
        """
        self.model = model
        self.train_loader = train_loader
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.only_proj_layer = only_proj_layer
        self.classification_mode = classification_mode
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.is_variant_effect_prediction = is_variant_effect_prediction
        self.cache_size = cache_size
        self.task_type = task_type
        self.num_bins = num_bins
        self.device = device
        if task_type not in ("classification", "regression"):
            raise ValueError(f"Unsupported task_type: {task_type!r}")
        if task_type == "regression" and not only_proj_layer:
            raise ValueError(
                "Regression fine-tuning supports only a frozen backbone (only_proj_layer=True)"
            )
        
        # Create projection layer for classification
        # For variant effect tasks, input is concatenated embeddings (hidden_dim * 2)
        proj_input_dim = (
            self.hidden_dim * 2
            if is_variant_effect_prediction
            else self.hidden_dim
        )
        self.projection = nn.Linear(proj_input_dim, self.num_outputs).to(device)
    
    def _get_optimizer(self, params):
        """
        Create optimizer based on configuration.
        
        Args:
            params: model parameters to optimize
            
        Returns:
            torch.optim.Optimizer: configured optimizer
        """
        optimizer_name = self.optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from: 'adam', 'adamw', 'sgd'")
            
    
    def _classification_loss(self, batch, criterion, fwd_cache):
        """Compute the classification loss for a batch."""
        if self.is_variant_effect_prediction:
            # Variant effect task: (variant_seqs, ref_seqs, labels, conditional_input)
            variant_sequences, ref_sequences, labels, conditional_input = batch
            labels = labels.to(self.device)
            
            # Get representative embeddings for both sequences
            # Use no_grad when only training projection layer (saves memory/compute)
            if self.only_proj_layer:
                with torch.no_grad():
                    var_repr = fwd_cache.cached_call(
                        self.model._sequence_to_representative,
                        variant_sequences,
                        disable=self.cache_size == 0,
                    )
                    ref_repr = fwd_cache.cached_call(
                        self.model._sequence_to_representative,
                        ref_sequences,
                        disable=self.cache_size == 0,
                    )
                # Detach to ensure no gradient flow to model
                var_repr = var_repr.detach()
                ref_repr = ref_repr.detach()
            else:
                var_repr = self.model._sequence_to_representative(variant_sequences)
                ref_repr = self.model._sequence_to_representative(ref_sequences)
            
            # Concatenate variant and reference representations
            sequence_repr = torch.cat([var_repr, ref_repr], dim=1)
        else:
            # Single sequence task: (sequences, labels, conditional_input)
            sequences, labels, conditional_input = batch
            labels = labels.to(self.device)
            
            # Get representative embeddings
            # Use no_grad when only training projection layer (saves memory/compute)
            if self.only_proj_layer:
                with torch.no_grad():
                    sequence_repr = fwd_cache.cached_call(
                        self.model._sequence_to_representative,
                        sequences,
                        disable=self.cache_size == 0,
                    )
                # Detach to ensure no gradient flow to model
                sequence_repr = sequence_repr.detach()
            else:
                sequence_repr = self.model._sequence_to_representative(sequences)

        if self.num_labels > 1 and self.num_classes == 2:
            labels = labels.float()
        
        logits = self.projection(sequence_repr)

        if self.num_labels > 1 and self.num_classes > 2:
            logits = logits.reshape(-1, self.num_labels, self.num_classes)
            loss = criterion(logits.transpose(1, 2), labels.long())
        else:
            loss = criterion(logits, labels)
        return loss

    def _regression_loss(self, batch, criterion):
        """Compute the MSE loss of the regression head on a batch.

        Features come from the frozen backbone's public infer_sequence_to_sequence.
        Binned tasks apply the head per token and pool the outputs into bins, which is
        what GFMWithProjection and the task do at evaluation.
        """
        sequences, labels, conditional_input = batch
        labels = labels.to(self.device).float()
        with torch.no_grad():
            _, token_embeddings, sequence_repr = self.model.infer_sequence_to_sequence(
                sequences, conditional_input
            )
        if self.num_bins == 1:
            features = sequence_repr
        else:
            features = token_embeddings
        if features is None:
            raise ValueError(
                "The model did not return the embeddings required for regression fine-tuning"
            )
        features = to_float_tensor(features, self.device)
        preds = self.projection(features)
        if self.num_bins > 1:
            valid_counts = get_valid_token_counts(self.model, list(sequences), preds.shape[1])
            preds = pool_tokens_to_bins(preds, valid_counts, self.num_bins)
        return criterion(preds, labels)

    def fine_tune(self):
        """
        Fine-tune the model or only the projection layer for classification tasks.
        
        Returns:
            GFMWithProjection: wrapped model with trained projection layer
        """
        # Determine which parameters to optimize
        if self.only_proj_layer:
            params = self.projection.parameters()
        else:
            params = list(self.model.parameters()) + list(self.projection.parameters())
        
        # Initialize optimizer
        optimizer = self._get_optimizer(params)
        
        if self.task_type == "regression":
            criterion = torch.nn.MSELoss()
        elif self.num_labels > 1 and self.num_classes == 2:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Set to training mode
        if not self.only_proj_layer:
            self.model.train()
        else:
            self.model.eval()  # Keep model in eval mode if only training projection
        self.projection.train()

        # Regression reads per-token embeddings, which are too large to cache.
        use_fwd_cache = self.only_proj_layer and self.task_type != "regression"
        fwd_cache = SequenceInferenceCache(max_size_gb=self.cache_size) if use_fwd_cache else None

        # Training loop
        avg_loss = float("nan")
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Fine-tuning epoch {epoch+1}/{self.num_epochs}")
            for batch in progress_bar:
                if self.task_type == "regression":
                    loss = self._regression_loss(batch, criterion)
                else:
                    loss = self._classification_loss(batch, criterion, fwd_cache)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update loss tracking
                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / num_batches
                
                # Update progress bar with average loss
                progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

        if fwd_cache is not None:
            fwd_cache.clear()

        if self.num_epochs > 0:
            print(f"Fine-tuning completed. Final average loss: {avg_loss:.4f}")
        else:
            print("Fine-tuning skipped")
        
        # Return wrapped model with projection layer
        wrapped_model = GFMWithProjection(
            self.model,
            self.projection,
            num_labels=self.num_labels,
            num_classes=self.num_classes,
            cache_size=self.cache_size,
            task_type=self.task_type,
            num_bins=self.num_bins,
        )
        wrapped_model.eval()  # Set to eval mode after training
        return wrapped_model


