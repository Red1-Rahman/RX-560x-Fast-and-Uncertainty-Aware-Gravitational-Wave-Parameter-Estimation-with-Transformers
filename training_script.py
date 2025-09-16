import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from test_transformer import GWTransformer
from dataset_gen import GWDatasetGenerator

class GWDataset(Dataset):
    """PyTorch Dataset for gravitational wave data"""
    
    def __init__(self, waveforms, parameters, normalize=True):
        """
        Parameters:
        -----------
        waveforms : np.array
            Waveform time series data
        parameters : dict
            Dictionary with parameter arrays
        normalize : bool
            Whether to normalize parameters for training
        """
        self.waveforms = torch.FloatTensor(waveforms)
        
        # Extract target parameters and normalize
        self.chirp_mass = torch.FloatTensor(parameters['chirp_mass'])
        self.mass_ratio = torch.FloatTensor(parameters['mass_ratio']) 
        self.tc_frac = torch.FloatTensor(parameters['tc_frac'])
        
        # Parameter normalization for stable training
        if normalize:
            # Chirp mass: normalize to [0, 1] range
            self.mc_min, self.mc_max = self.chirp_mass.min(), self.chirp_mass.max()
            self.chirp_mass = (self.chirp_mass - self.mc_min) / (self.mc_max - self.mc_min)
            
            # Mass ratio already in [0, 1]
            # tc_frac already in [0, 1]
        
        # Stack parameters
        self.targets = torch.stack([self.chirp_mass, self.mass_ratio, self.tc_frac], dim=1)
        
        print(f"Dataset created:")
        print(f"  Waveforms: {self.waveforms.shape}")
        print(f"  Targets: {self.targets.shape}")
        if normalize:
            print(f"  Chirp mass range: [{self.mc_min:.1f}, {self.mc_max:.1f}] M☉")
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        return self.waveforms[idx], self.targets[idx]

class GaussianNLLLoss(nn.Module):
    """Gaussian Negative Log-Likelihood loss for uncertainty-aware training"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions, log_var, targets):
        # Gaussian NLL: 0.5 * (log_var + (pred - target)^2 / var)
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + (predictions - targets)**2 / var)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for waveforms, targets in pbar:
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, log_var = model(waveforms)
        
        # Calculate loss
        loss = criterion(predictions, log_var, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    predictions_list = []
    targets_list = []
    uncertainties_list = []
    
    with torch.no_grad():
        for waveforms, targets in dataloader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions, log_var = model(waveforms)
            
            # Calculate loss
            loss = criterion(predictions, log_var, targets)
            total_loss += loss.item()
            
            # Store predictions for analysis
            predictions_list.append(predictions.cpu())
            targets_list.append(targets.cpu())
            uncertainties_list.append(torch.exp(0.5 * log_var).cpu())  # Standard deviation
    
    # Concatenate results
    all_predictions = torch.cat(predictions_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    all_uncertainties = torch.cat(uncertainties_list, dim=0)
    
    return total_loss / len(dataloader), all_predictions, all_targets, all_uncertainties

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss') 
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('Training History (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_predictions(predictions, targets, uncertainties, param_names=['Chirp Mass', 'Mass Ratio', 'Tc Fraction']):
    """Plot prediction vs target scatter plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (param_name, ax) in enumerate(zip(param_names, axes)):
        pred = predictions[:, i].numpy()
        true = targets[:, i].numpy()
        unc = uncertainties[:, i].numpy()
        
        # Scatter plot with error bars
        ax.errorbar(true, pred, yerr=unc, fmt='o', alpha=0.6, markersize=3)
        
        # Perfect prediction line
        min_val, max_val = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ax.set_xlabel(f'True {param_name}')
        ax.set_ylabel(f'Predicted {param_name}')
        ax.set_title(f'{param_name} Predictions')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((pred - true)**2))
        ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Training configuration
    config = {
        'batch_size': 16,       # Small batch for CPU
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cpu',        # Change to 'cuda' when using 3060 Ti
        'save_path': 'gw_transformer.pth'
    }
    
    print("Gravitational Wave Parameter Estimation Training")
    print("=" * 50)
    
    # Generate training dataset
    print("Generating training dataset...")
    generator = GWDatasetGenerator(duration=0.25)  # 0.25s for CPU development
    
    # Generate train and validation sets
    train_data = generator.generate_dataset(n_samples=800, save_path='train_dataset.pkl')
    val_data = generator.generate_dataset(n_samples=200, save_path='val_dataset.pkl')
    
    # Create PyTorch datasets
    train_dataset = GWDataset(train_data['waveforms'], train_data['parameters'])
    val_dataset = GWDataset(val_data['waveforms'], val_data['parameters'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = GWTransformer(seq_len=1024, d_model=64, nhead=4, num_layers=2)
    
    device = torch.device(config['device'])
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {device}")
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = GaussianNLLLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, predictions, targets, uncertainties = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['save_path'])
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot results
    plot_training_history(train_losses, val_losses)
    plot_predictions(predictions, targets, uncertainties)
    
    print(f"Model saved as '{config['save_path']}'")
    print("Training plots saved as 'training_history.png' and 'prediction_results.png'")

if __name__ == "__main__":
    main()