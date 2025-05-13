"""
Axial Attention Trainer Implementation

This module provides a trainer for Axial Attention models using PyTorch
as the backend. It handles training, evaluation, and model management
for the accurate path in the trading system.
"""

import numpy as np
from typing import Dict, List, Any, Optional

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except (ImportError, ValueError, AttributeError) as e:
    HAS_PYTORCH = False
    import logging
    logging.warning(f"PyTorch not available or incompatible: {str(e)}. Axial Attention training will not function.")

from src.ml.trainer.base_trainer import ModelTrainer

class AxialAttentionModel(nn.Module):
    """
    PyTorch implementation of Axial Attention model.
    
    This model uses axial attention mechanisms to process market data and
    generate trading signals.
    """
    
    def __init__(self, seq_length: int, num_features: int, num_heads: int, head_dim: int, 
                num_layers: int, dropout: float = 0.1, num_classes: int = 3):
        """
        Initialize the Axial Attention model.
        
        Args:
            seq_length: Length of input sequences
            num_features: Number of features per time step
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Input embedding
        self.embedding = nn.Linear(num_features, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_length, self.hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            self._create_transformer_layer() for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(self.hidden_dim, num_classes)
        
    def _create_transformer_layer(self) -> nn.Module:
        """
        Create a transformer layer with axial attention.
        
        Returns:
            Transformer layer module
        """
        return nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, num_features]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.output_norm(x)
        x = self.output_dropout(x)
        x = self.output_linear(x)
        
        return x

class AxialAttentionTrainer(ModelTrainer):
    """
    Trainer for Axial Attention models.
    
    This trainer uses PyTorch to train Axial Attention models for the accurate path
    in the trading system. It handles data preparation, training, evaluation,
    and model management.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Axial Attention trainer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        super().__init__("axial_attention", config)
        
        # Check if PyTorch is available
        if not HAS_PYTORCH:
            self.logger.error("PyTorch is not available. Axial Attention training will not function.")
            raise ImportError("PyTorch is required for Axial Attention training")
        
        # Initialize model parameters from config
        self.num_heads = self.model_config.get("num_heads", 8)
        self.head_dim = self.model_config.get("head_dim", 64)
        self.num_layers = self.model_config.get("num_layers", 6)
        self.seq_length = self.model_config.get("seq_length", 100)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.hidden_dim = self.model_config.get("hidden_dim", 512)
        self.num_epochs = self.model_config.get("num_epochs", 100)
        self.learning_rate = self.model_config.get("learning_rate", 0.0005)
        self.patience = self.model_config.get("patience", 15)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.num_classes = self.model_config.get("output_size", 3)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Initialized Axial Attention trainer with parameters: "
                       f"heads={self.num_heads}, "
                       f"head_dim={self.head_dim}, "
                       f"layers={self.num_layers}, "
                       f"seq_length={self.seq_length}, "
                       f"device={self.device}")
    
    def build_model(self) -> nn.Module:
        """
        Build the Axial Attention model architecture.
        
        Returns:
            PyTorch model
        """
        self.logger.info("Building Axial Attention model")
        
        # Create model
        model = AxialAttentionModel(
            seq_length=self.seq_length,
            num_features=6,  # OHLCV + VWAP
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes
        )
        
        # Move model to device
        model = model.to(self.device)
        
        self.logger.info(f"Built Axial Attention model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def train(self, train_data: Dict[str, Any], validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the Axial Attention model on the provided data.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Training history/metrics
        """
        self.logger.info("Training Axial Attention model")
        
        # Start timing
        self.latency_profiler.start_phase("training")
        
        # Process training data
        train_loader = self._prepare_data_loader(train_data)
        
        # Process validation data if provided
        valid_loader = None
        if validation_data is not None:
            valid_loader = self._prepare_data_loader(validation_data)
            
        # Check if model is built
        if self.model is None:
            self.model = self.build_model()
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Set up early stopping
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": []
        }
        
        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{self.num_epochs} | "
                                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                                    f"Loss: {loss.item():.4f}")
            
            # Calculate training metrics
            train_loss /= len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            
            # Validation phase
            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            
            if valid_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        # Move data to device
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Update statistics
                        valid_loss += loss.item()
                        _, predicted = outputs.max(1)
                        valid_total += targets.size(0)
                        valid_correct += predicted.eq(targets).sum().item()
                
                # Calculate validation metrics
                valid_loss /= len(valid_loader)
                valid_accuracy = 100.0 * valid_correct / valid_total
                
                # Check for early stopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Update history
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            
            if valid_loader is not None:
                history["valid_loss"].append(valid_loss)
                history["valid_accuracy"].append(valid_accuracy)
                
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                               f"Train Loss: {train_loss:.4f} | "
                               f"Train Acc: {train_accuracy:.2f}% | "
                               f"Valid Loss: {valid_loss:.4f} | "
                               f"Valid Acc: {valid_accuracy:.2f}%")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                               f"Train Loss: {train_loss:.4f} | "
                               f"Train Acc: {train_accuracy:.2f}%")
        
        # Load best model if early stopping occurred
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # End timing
        training_time = self.latency_profiler.end_phase()
        
        # Add training time to history
        history["training_time_ms"] = training_time
        
        self.logger.info(f"Completed Axial Attention training in {training_time:.2f} ms")
        
        return history
    
    def _prepare_data_loader(self, data: Dict[str, Any]) -> DataLoader:
        """
        Prepare PyTorch DataLoader from training data.
        
        Args:
            data: Training data dictionary
            
        Returns:
            PyTorch DataLoader
        """
        # Extract sequences and labels
        all_sequences = []
        all_labels = []
        
        for symbol, sequences in data.get("sequences", {}).items():
            labels = data.get("labels", {}).get(symbol, [])
            
            if len(sequences) != len(labels):
                self.logger.warning(f"Mismatch between sequences and labels for {symbol}")
                continue
                
            all_sequences.append(sequences)
            all_labels.append(labels)
        
        if not all_sequences:
            raise ValueError("No valid sequences found in data")
            
        # Concatenate data from all symbols
        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)  # Convert one-hot to class indices
        
        # Create dataset and data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        return loader
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating Axial Attention model")
        
        # Start timing
        self.latency_profiler.start_phase("evaluation")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Process test data
        test_loader = self._prepare_data_loader(test_data)
        
        # Set up loss function
        criterion = nn.CrossEntropyLoss()
        
        # Evaluation phase
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                # Store targets and predictions for additional metrics
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Calculate test metrics
        test_loss /= len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_predictions).tolist()
        
        # End timing
        evaluation_time = self.latency_profiler.end_phase()
        
        # Create metrics dictionary
        metrics = {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
            "evaluation_time_ms": evaluation_time
        }
        
        self.logger.info(f"Completed Axial Attention evaluation in {evaluation_time:.2f} ms "
                       f"with metrics: loss={test_loss:.4f}, accuracy={test_accuracy:.2f}%")
        
        return metrics
    
    def save(self, version: str) -> str:
        """
        Save the trained model with the specified version.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            Path to saved model
        """
        self.logger.info(f"Saving Axial Attention model version {version}")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Save model using ModelVersionManager
        model_dir = self.version_manager.save_model(
            "axial_attention",
            version,
            self.model,
            self.training_metadata,
            self.performance_metrics,
            self.latency_profiler.create_latency_profile()
        )
        
        self.logger.info(f"Saved Axial Attention model version {version} to {model_dir}")
        
        return model_dir
    
    def load(self, version: Optional[str] = None) -> None:
        """
        Load a trained model.
        
        Args:
            version: Version to load (optional, defaults to active version)
        """
        self.logger.info(f"Loading Axial Attention model version {version if version else 'active'}")
        
        # Load model using ModelVersionManager
        model, metadata = self.version_manager.load_model("axial_attention", version)
        
        # Set model and metadata
        self.model = model
        self.training_metadata = metadata
        
        # Move model to device
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        
        self.logger.info(f"Loaded Axial Attention model version {version if version else 'active'}")
    
    def _generate_benchmark_data(self, num_samples: int) -> List[Any]:
        """
        Generate synthetic data for benchmarking.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of benchmark data samples
        """
        # Generate random sequences
        X = np.random.rand(num_samples, self.seq_length, 6)
        
        return X
    
    def _benchmark_inference_batch(self, batch: List[Any]) -> List[Any]:
        """
        Run inference on a batch of data for benchmarking.
        
        Args:
            batch: Batch of data
            
        Returns:
            Inference results
        """
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Convert batch to PyTorch tensor
        X = torch.tensor(batch, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(X)
            _, predictions = outputs.max(1)
        
        return predictions.cpu().numpy()
