import torch
import torch.nn as nn
from tqdm import tqdm

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
        num_labels,
        num_epochs,
        lr,
        optimizer_name="AdamW",
        weight_decay=0.01,
        only_proj_layer=True,
        is_variant_effect_prediction=False,
        device='cpu'
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model: BaseGFMModel instance
            train_loader: DataLoader for training data
            hidden_dim: hidden dimension size of the model
            num_labels: number of classification labels
            num_epochs: number of training epochs
            lr: learning rate
            optimizer_name: optimizer name ('Adam', 'AdamW', 'SGD')
            weight_decay: weight decay for regularization
            only_proj_layer: if True, only train projection layer; if False, train full model
            is_variant_effect_prediction: if True, task uses variant/ref sequence pairs
            device: torch device
        """
        self.model = model
        self.train_loader = train_loader
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.only_proj_layer = only_proj_layer
        self.is_variant_effect_prediction = is_variant_effect_prediction
        self.device = device
        
        # Create projection layer for classification
        # For variant effect tasks, input is concatenated embeddings (hidden_dim * 2)
        proj_input_dim = self.hidden_dim * 2 if is_variant_effect_prediction else self.hidden_dim
        self.projection = nn.Linear(proj_input_dim, self.num_labels).to(device)
    
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
        
        # Loss function for classification
        criterion = torch.nn.CrossEntropyLoss()
        
        # Set to training mode
        if not self.only_proj_layer:
            self.model.train()
        else:
            self.model.eval()  # Keep model in eval mode if only training projection
        self.projection.train()
        
        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Fine-tuning epoch {epoch+1}/{self.num_epochs}")
            for batch in progress_bar:
                if self.is_variant_effect_prediction:
                    # Variant effect task: (variant_seqs, ref_seqs, labels, conditional_input)
                    variant_sequences, ref_sequences, labels, conditional_input = batch
                    labels = labels.to(self.device)
                    
                    # Get representative embeddings for both sequences
                    # Use no_grad when only training projection layer (saves memory/compute)
                    if self.only_proj_layer:
                        with torch.no_grad():
                            var_repr = self.model._sequence_to_representative(variant_sequences)
                            ref_repr = self.model._sequence_to_representative(ref_sequences)
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
                            sequence_repr = self.model._sequence_to_representative(sequences)
                        # Detach to ensure no gradient flow to model
                        sequence_repr = sequence_repr.detach()
                    else:
                        sequence_repr = self.model._sequence_to_representative(sequences)
                
                logits = self.projection(sequence_repr)
                
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update loss tracking
                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / num_batches
                
                # Update progress bar with average loss
                progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        if self.num_epochs > 0:
            print(f"Fine-tuning completed. Final average loss: {avg_loss:.4f}")
        else:
            print("Fine-tuning skipped")
        
        # Return wrapped model with projection layer
        wrapped_model = GFMWithProjection(self.model, self.projection)
        wrapped_model.eval()  # Set to eval mode after training
        return wrapped_model


