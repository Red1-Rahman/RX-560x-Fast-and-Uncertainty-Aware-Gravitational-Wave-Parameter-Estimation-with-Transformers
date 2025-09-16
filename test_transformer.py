import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series data
    """
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GWTransformer(nn.Module):
    """
    Transformer model for gravitational wave parameter estimation
    """
    def __init__(self, 
                 seq_len=1024,
                 d_model=64, 
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 n_params=3):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_params = n_params
        
        # Input projection: 1D time series -> d_model embeddings
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Parameter regression heads
        self.param_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, n_params)
        )
        
        # Uncertainty estimation head (log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 4),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 4, n_params)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Reshape for linear layer: (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Input projection: (batch_size, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over time dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pooling(x)    # (batch_size, d_model, 1)
        x = x.squeeze(-1)      # (batch_size, d_model)
        
        # Parameter prediction
        params = self.param_head(x)
        
        # Uncertainty estimation
        log_var = self.uncertainty_head(x)
        
        return params, log_var

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing GW Transformer model...")
    
    # CPU-friendly model configuration
    model = GWTransformer(
        seq_len=1024,       # 0.25s @ 4096 Hz
        d_model=64,         # Small embedding for CPU
        nhead=4,            # 4 attention heads
        num_layers=2,       # 2 transformer layers
        dim_feedforward=256,
        dropout=0.1,
        n_params=3          # chirp_mass, mass_ratio, tc_frac
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model size: {count_parameters(model) * 4 / 1e6:.1f} MB (float32)")
    
    # Test forward pass
    batch_size = 4
    seq_len = 1024
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: ({batch_size}, {seq_len})")
    
    # Random input (simulating waveforms)
    x = torch.randn(batch_size, seq_len)
    
    # Forward pass
    try:
        with torch.no_grad():
            params, log_var = model(x)
        
        print(f"✓ Forward pass successful!")
        print(f"Output params shape: {params.shape}")
        print(f"Output log_var shape: {log_var.shape}")
        
        # Print some example outputs
        print(f"\nExample predictions:")
        print(f"Params: {params[0].numpy()}")
        print(f"Log variance: {log_var[0].numpy()}")
        
        print(f"\n✓ Model test passed! Ready for training.")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")