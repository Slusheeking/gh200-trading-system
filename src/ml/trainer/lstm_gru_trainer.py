"""
LSTM/GRU Trainer Implementation

This module provides a trainer for LSTM/GRU models using PyTorch
as the backend. It handles training, evaluation, and model management
for the exit optimization in the trading system.
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
    logging.warning(f"PyTorch not available or incompatible: {str(e)}. LSTM/GRU training will not function.")

from src.ml.trainer.base_trainer import ModelTrainer

class LSTMGRUModel(nn.Module):
    """
    PyTorch implementation of LSTM/GRU model for exit optimization.
    
    This model uses LSTM/GRU architecture to analyze market data and active positions
    to determine the optimal exit points.
    """
    
    def __init__(self, seq_length: int, num_features: int, hidden_size: int, 
                num_layers: int, bidirectional: bool = True, use_gru: bool = False,
                attention_enabled: bool = True, dropout: float = 0.1, output_size: int = 3):
        """
        Initialize the LSTM/GRU model.
        
        Args:
            seq_length: Length of input sequences
            num_features: Number of features per time step
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent layers
            bidirectional: Whether to use bidirectional LSTM/GRU
            use_gru: Whether to use GRU instead of LSTM
            attention_enabled: Whether to use attention mechanism
            dropout: Dropout rate
            output_size: Number of output values
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_gru = use_gru
        self.attention_enabled = attention_enabled
        self.dropout = dropout
        self.output_size = output_size
        
        # Calculate dimensions
        self.num_directions = 2 if bidirectional else 1
        self.rnn_output_size = hidden_size * self.num_directions
        
        # Input embedding
        self.embedding = nn.Linear(num_features, hidden_size)
        
        # Recurrent layer (LSTM or GRU)
        if use_gru:
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Attention mechanism
        if attention_enabled:
            self.attention = nn.Sequential(
                nn.Linear(self.rnn_output_size, self.rnn_output_size),
                nn.Tanh(),
                nn.Linear(self.rnn_output_size, 1)
            )
        
        # Output layers
        self.output_dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(self.rnn_output_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, num_features]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Input embedding
        x = self.embedding(x)
        
        # Recurrent layer
        rnn_output, _ = self.rnn(x)
        
        # Apply attention if enabled
        if self.attention_enabled:
            # Calculate attention weights
            attention_weights = self.attention(rnn_output)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Apply attention weights
            context = torch.bmm(attention_weights.transpose(1, 2), rnn_output)
            output = context.squeeze(1)
        else:
            # Use last output
            output = rnn_output[:, -1, :]
        
        # Output layers
        output = self.output_dropout(output)
        output = self.output_linear(output)
        
        return output

class LSTMGRUTrainer(ModelTrainer):
    """
    Trainer for LSTM/GRU models.
    
    This trainer uses PyTorch to train LSTM/GRU models for exit optimization
    in the trading system. It handles data preparation, training, evaluation,
    and model management.
    """
    
    def __init__(self, config=None):
        """
        Initialize the LSTM/GRU trainer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        super().__init__("lstm_gru", config)
        
        # Check if PyTorch is available
        if not HAS_PYTORCH:
            self.logger.error("PyTorch is not available. LSTM/GRU training will not function.")
            raise ImportError("PyTorch is required for LSTM/GRU training")
        
        # Initialize model parameters from config
        self.num_layers = self.model_config.get("num_layers", 3)
        self.hidden_size = self.model_config.get("hidden_size", 128)
        self.bidirectional = self.model_config.get("bidirectional", True)
        self.use_gru = self.model_config.get("use_gru", False)
        self.attention_enabled = self.model_config.get("attention_enabled", True)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.seq_length = self.model_config.get("seq_length", 50)
        self.num_epochs = self.model_config.get("num_epochs", 100)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.patience = self.model_config.get("patience", 15)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.output_size = self.model_config.get("output_size", 3)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Initialized LSTM/GRU trainer with parameters: "
                       f"layers={self.num_layers}, "
                       f"hidden_size={self.hidden_size}, "
                       f"bidirectional={self.bidirectional}, "
                       f"use_gru={self.use_gru}, "
                       f"attention={self.attention_enabled}, "
                       f"seq_length={self.seq_length}, "
                       f"device={self.device}")
    
    def build_model(self) -> nn.Module:
        """
        Build the LSTM/GRU model architecture.
        
        Returns:
            PyTorch model
        """
        self.logger.info("Building LSTM/GRU model")
        
        # Create model
        model = LSTMGRUModel(
            seq_length=self.seq_length,
            num_features=6,  # OHLCV + VWAP
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            use_gru=self.use_gru,
            attention_enabled=self.attention_enabled,
            dropout=self.dropout,
            output_size=self.output_size
        )
        
        # Move model to device
        model = model.to(self.device)
        
        self.logger.info(f"Built {'GRU' if self.use_gru else 'LSTM'} model with "
                       f"{sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def train(self, train_data: Dict[str, Any], validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the LSTM/GRU model on the provided data.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Training history/metrics
        """
        self.logger.info("Training LSTM/GRU model")
        
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
        criterion = nn.MSELoss()
        
        # Set up early stopping
        best_valid_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            "train_loss": [],
            "valid_loss": []
        }
        
        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
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
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{self.num_epochs} | "
                                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                                    f"Loss: {loss.item():.4f}")
            
            # Calculate training metrics
            train_loss /= len(train_loader)
            
            # Validation phase
            valid_loss = 0.0
            
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
                
                # Calculate validation metrics
                valid_loss /= len(valid_loader)
                
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
            
            if valid_loader is not None:
                history["valid_loss"].append(valid_loss)
                
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                               f"Train Loss: {train_loss:.4f} | "
                               f"Valid Loss: {valid_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                               f"Train Loss: {train_loss:.4f}")
        
        # Load best model if early stopping occurred
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # End timing
        training_time = self.latency_profiler.end_phase()
        
        # Add training time to history
        history["training_time_ms"] = training_time
        
        self.logger.info(f"Completed LSTM/GRU training in {training_time:.2f} ms")
        
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
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
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
        self.logger.info("Evaluating LSTM/GRU model")
        
        # Start timing
        self.latency_profiler.start_phase("evaluation")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Process test data
        test_loader = self._prepare_data_loader(test_data)
        
        # Set up loss function
        criterion = nn.MSELoss()
        
        # Evaluation phase
        self.model.eval()
        test_loss = 0.0
        
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
                
                # Store targets and predictions for additional metrics
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
        
        # Calculate test metrics
        test_loss /= len(test_loader)
        
        # Concatenate all targets and predictions
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate additional metrics
        mse = np.mean((all_targets - all_predictions) ** 2, axis=0)
        mae = np.mean(np.abs(all_targets - all_predictions), axis=0)
        
        # End timing
        evaluation_time = self.latency_profiler.end_phase()
        
        # Create metrics dictionary
        metrics = {
            "loss": test_loss,
            "mse": mse.tolist(),
            "mae": mae.tolist(),
            "evaluation_time_ms": evaluation_time
        }
        
        self.logger.info(f"Completed LSTM/GRU evaluation in {evaluation_time:.2f} ms "
                       f"with metrics: loss={test_loss:.4f}")
        
        return metrics
    
    def save(self, version: str) -> str:
        """
        Save the trained model with the specified version.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            Path to saved model
        """
        self.logger.info(f"Saving LSTM/GRU model version {version}")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Save model using ModelVersionManager
        model_dir = self.version_manager.save_model(
            "lstm_gru",
            version,
            self.model,
            self.training_metadata,
            self.performance_metrics,
            self.latency_profiler.create_latency_profile()
        )
        
        self.logger.info(f"Saved LSTM/GRU model version {version} to {model_dir}")
        
        return model_dir
    
    def load(self, version: Optional[str] = None) -> None:
        """
        Load a trained model.
        
        Args:
            version: Version to load (optional, defaults to active version)
        """
        self.logger.info(f"Loading LSTM/GRU model version {version if version else 'active'}")
        
        # Load model using ModelVersionManager
        model, metadata = self.version_manager.load_model("lstm_gru", version)
        
        # Set model and metadata
        self.model = model
        self.training_metadata = metadata
        
        # Move model to device
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        
        self.logger.info(f"Loaded LSTM/GRU model version {version if version else 'active'}")
    
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
        
        return outputs.cpu().numpy()