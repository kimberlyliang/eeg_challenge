#!/usr/bin/env python3
"""
Demo script for EEG Challenge 1 Models

This script demonstrates the model implementations and basic data analysis
without requiring the full dataset to be downloaded.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EEGNeX(nn.Module):
    """EEGNeX model implementation based on braindecode"""
    
    def __init__(self, n_chans=129, n_outputs=1, n_times=200, sfreq=100):
        super(EEGNeX, self).__init__()
        
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(n_chans, 32, kernel_size=15, padding=7)
        self.temporal_bn = nn.BatchNorm1d(32)
        self.temporal_activation = nn.ELU()
        self.temporal_pool = nn.AvgPool1d(kernel_size=2)
        
        # Spatial convolution
        self.spatial_conv = nn.Conv1d(32, 64, kernel_size=1)
        self.spatial_bn = nn.BatchNorm1d(64)
        self.spatial_activation = nn.ELU()
        
        # Feature extraction
        self.feature_conv1 = nn.Conv1d(64, 128, kernel_size=10, padding=4)
        self.feature_bn1 = nn.BatchNorm1d(128)
        self.feature_activation1 = nn.ELU()
        self.feature_pool1 = nn.AvgPool1d(kernel_size=2)
        
        self.feature_conv2 = nn.Conv1d(128, 256, kernel_size=10, padding=4)
        self.feature_bn2 = nn.BatchNorm1d(256)
        self.feature_activation2 = nn.ELU()
        self.feature_pool2 = nn.AvgPool1d(kernel_size=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs)
        )
        
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.temporal_activation(x)
        x = self.temporal_pool(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_activation(x)
        
        # Feature extraction
        x = self.feature_conv1(x)
        x = self.feature_bn1(x)
        x = self.feature_activation1(x)
        x = self.feature_pool1(x)
        
        x = self.feature_conv2(x)
        x = self.feature_bn2(x)
        x = self.feature_activation2(x)
        x = self.feature_pool2(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class SimpleEEGNet(nn.Module):
    """Simplified EEGNet for comparison"""
    
    def __init__(self, n_chans=129, n_outputs=1, n_times=200, sfreq=100):
        super(SimpleEEGNet, self).__init__()
        
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.sfreq = sfreq
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(1, 8, (1, 64), padding=(0, 32))
        self.temporal_bn = nn.BatchNorm2d(8)
        
        # Spatial convolution
        self.spatial_conv = nn.Conv2d(8, 16, (n_chans, 1), bias=False)
        self.spatial_bn = nn.BatchNorm2d(16)
        self.spatial_activation = nn.ELU()
        self.spatial_pool = nn.AvgPool2d((1, 4))
        
        # Feature extraction
        self.feature_conv = nn.Conv2d(16, 32, (1, 16), padding=(0, 8))
        self.feature_bn = nn.BatchNorm2d(32)
        self.feature_activation = nn.ELU()
        self.feature_pool = nn.AvgPool2d((1, 8))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * (n_times // 32), 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_outputs)
        )
        
    def forward(self, x):
        # x shape: (batch, n_chans, n_times)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, n_chans, n_times)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        
        # Spatial convolution
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.spatial_activation(x)
        x = self.spatial_pool(x)
        
        # Feature extraction
        x = self.feature_conv(x)
        x = self.feature_bn(x)
        x = self.feature_activation(x)
        x = self.feature_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CNN1D(nn.Module):
    """Simple 1D CNN for comparison"""
    
    def __init__(self, n_chans=129, n_outputs=1, n_times=200):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(n_chans, 64, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions
        self.fc_input_size = 256 * (n_times // 8)  # After 3 pooling operations
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs)
        )
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_synthetic_data(n_samples=1000, n_chans=129, n_times=200):
    """Create synthetic EEG-like data for demonstration"""
    print("Creating synthetic EEG data...")
    
    # Generate synthetic EEG-like data
    eeg_data = np.random.randn(n_samples, n_chans, n_times) * 0.1
    
    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add alpha rhythm (8-12 Hz)
        t = np.linspace(0, 2, n_times)  # 2 seconds
        alpha = 0.05 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
        eeg_data[i, :, :] += alpha
        
        # Add some channel-specific patterns
        for ch in range(n_chans):
            if ch % 10 == 0:  # Every 10th channel
                eeg_data[i, ch, :] += 0.02 * np.sin(2 * np.pi * 5 * t)
    
    # Generate response times with some correlation to the data
    # Simulate that higher alpha power leads to faster responses
    alpha_power = np.mean(eeg_data[:, :, 50:150], axis=(1, 2))  # Alpha band power
    response_times = 0.5 + 0.3 * np.exp(-alpha_power) + np.random.normal(0, 0.1, n_samples)
    response_times = np.clip(response_times, 0.2, 2.0)  # Clip to realistic range
    
    return eeg_data, response_times

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    """Train a model with simple training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Training {model.__class__.__name__}...")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test).squeeze().cpu().numpy()
        targets = y_test.cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    normalized_rmse = rmse / np.std(targets)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'normalized_rmse': normalized_rmse,
        'predictions': predictions,
        'targets': targets
    }

def plot_results(results, train_losses, val_losses):
    """Plot training results and predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Predictions vs targets
    axes[0, 1].scatter(results['targets'], results['predictions'], alpha=0.6)
    axes[0, 1].plot([results['targets'].min(), results['targets'].max()], 
                    [results['targets'].min(), results['targets'].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Response Time (s)')
    axes[0, 1].set_ylabel('Predicted Response Time (s)')
    axes[0, 1].set_title(f'Predictions vs Targets (R² = {results["r2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = results['predictions'] - results['targets']
    axes[1, 0].scatter(results['targets'], residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('True Response Time (s)')
    axes[1, 0].set_ylabel('Residuals (s)')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of predictions and targets
    axes[1, 1].hist(results['targets'], alpha=0.7, label='True', bins=30, color='blue')
    axes[1, 1].hist(results['predictions'], alpha=0.7, label='Predicted', bins=30, color='red')
    axes[1, 1].set_xlabel('Response Time (s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demonstration function"""
    print("EEG Challenge 1: Model Demonstration")
    print("="*50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data
    print("\n1. Creating synthetic EEG data...")
    X, y = create_synthetic_data(n_samples=2000, n_chans=129, n_times=200)
    print(f"Data shape: {X.shape}, Response times shape: {y.shape}")
    print(f"Response time stats: mean={y.mean():.3f}s, std={y.std():.3f}s")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Define models
    models = {
        'EEGNeX': EEGNeX(n_chans=129, n_outputs=1, n_times=200),
        'SimpleEEGNet': SimpleEEGNet(n_chans=129, n_outputs=1, n_times=200),
        'CNN1D': CNN1D(n_chans=129, n_outputs=1, n_times=200)
    }
    
    # Train and evaluate models
    results = {}
    
    for model_name, model in models.items():
        print(f"\n2. Training {model_name}...")
        
        # Train model
        train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=50)
        
        # Evaluate model
        test_results = evaluate_model(model, X_test, y_test)
        results[model_name] = {
            'test_results': test_results,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"Test Results for {model_name}:")
        print(f"  RMSE: {test_results['rmse']:.4f}")
        print(f"  MAE: {test_results['mae']:.4f}")
        print(f"  R²: {test_results['r2']:.4f}")
        print(f"  Normalized RMSE: {test_results['normalized_rmse']:.4f}")
        
        # Plot results
        plot_results(test_results, train_losses, val_losses)
    
    # Compare models
    print(f"\n3. Model Comparison:")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Norm RMSE':<12}")
    print("-" * 60)
    for model_name, result in results.items():
        test_res = result['test_results']
        print(f"{model_name:<15} {test_res['rmse']:<10.4f} {test_res['mae']:<10.4f} "
              f"{test_res['r2']:<10.4f} {test_res['normalized_rmse']:<12.4f}")
    
    print(f"\n4. Key Insights:")
    print(f"- All models can learn to predict response times from EEG data")
    print(f"- The normalized RMSE metric is used for challenge evaluation")
    print(f"- Lower values indicate better performance")
    print(f"- In practice, you would use real EEG data from .bdf files")
    
    print(f"\n5. Next Steps:")
    print(f"- Implement actual EEG data loading from your dataset")
    print(f"- Add more sophisticated feature extraction")
    print(f"- Implement cross-task transfer learning")
    print(f"- Optimize hyperparameters for better performance")
    
    print(f"\nDemo complete!")

if __name__ == "__main__":
    main()

