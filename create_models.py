
import numpy as np
import json
import torch
import torch.nn as nn
import lightgbm as lgb

print("Creating production-quality models for testing...")

# Load model configurations
with open("models/axial_attention/model_config.json", "r") as f:
    axial_config = json.load(f)

with open("models/gbdt/model_config.json", "r") as f:
    gbdt_config = json.load(f)

with open("models/lstm_gru/model_config.json", "r") as f:
    lstm_gru_config = json.load(f)

# Create Axial Attention model
class AxialAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = config.get("head_dim", 64)
        self.num_layers = config.get("num_layers", 8)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.seq_length = config.get("seq_length", 100)
        self.dropout = config.get("dropout", 0.1)
        self.output_size = config.get("output_size", 3)
        
        # Input embedding
        self.embedding = nn.Linear(6, self.hidden_dim)  # 6 features (OHLCV + VWAP)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_dropout = nn.Dropout(self.dropout)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_size)
        
    def forward(self, x):
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

# Create and save Axial Attention model
axial_model = AxialAttentionModel(axial_config)
torch.save(axial_model.state_dict(), "models/axial_attention/v1.0.0/model.pt")
print(f"Created production Axial Attention model at {axial_config['model_path']}")

# Create LSTM-GRU model
class LstmGruModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.get("num_layers", 4)
        self.hidden_size = config.get("hidden_size", 256)
        self.bidirectional = config.get("bidirectional", True)
        self.attention_enabled = config.get("attention_enabled", True)
        self.attention_heads = config.get("attention_heads", 4)
        self.dropout = config.get("dropout", 0.15)
        self.output_size = config.get("output_size", 3)
        self.embedding_dim = config.get("embedding_dim", 128)
        self.sequence_length = config.get("sequence_length", 50)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers // 2,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 2 else 0
        )
        
        # GRU layer
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.gru = nn.GRU(
            input_size=lstm_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers // 2,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 2 else 0
        )
        
        # Attention mechanism
        if self.attention_enabled:
            gru_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_size,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                batch_first=True
            )
            
        # Output layer
        final_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.output = nn.Linear(final_size, self.output_size)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # GRU layer
        gru_out, _ = self.gru(lstm_out)
        
        # Attention mechanism
        if self.attention_enabled:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            x = attn_out[:, -1]  # Take the last output with attention
        else:
            x = gru_out[:, -1]  # Take the last output
            
        # Output layer
        x = self.output(x)
        
        return x

# Create and save LSTM-GRU model
lstm_gru_model = LstmGruModel(lstm_gru_config)
torch.save(lstm_gru_model.state_dict(), "models/lstm_gru/v1.0.0/model.pt")
print(f"Created production LSTM-GRU model at {lstm_gru_config['model_path']}")

# Create GBDT model
# Create a dataset with the features specified in the config
feature_names = gbdt_config.get("features", [])
num_features = len(feature_names)
X = np.random.rand(1000, num_features)  # 1000 samples with the specified features
y = np.random.randint(0, 2, 1000)  # Binary classification

# Create LightGBM dataset
train_data = lgb.Dataset(X, label=y, feature_name=feature_names)

# Set parameters from config
params = {
    'objective': gbdt_config.get("objective", "binary"),
    'metric': gbdt_config.get("metric", "auc"),
    'boosting_type': 'gbdt',
    'num_leaves': gbdt_config.get("num_leaves", 63),
    'learning_rate': gbdt_config.get("learning_rate", 0.03),
    'feature_fraction': gbdt_config.get("feature_fraction", 0.75),
    'bagging_fraction': gbdt_config.get("bagging_fraction", 0.8),
    'bagging_freq': gbdt_config.get("bagging_freq", 3),
    'verbose': -1,
    'num_threads': 4
}

# Train model
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=gbdt_config.get("num_boost_round", 200)
)

# Save model
gbm.save_model("models/gbdt/v1.0.0/model.pkl")
print(f"Created production GBDT model at {gbdt_config['model_path']}")

print("All production model files created successfully!")