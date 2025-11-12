#!/usr/bin/env python3
"""
EEG Challenge 1: Cross-Task Transfer Learning Implementation

This script implements the complete pipeline for Challenge 1 including:
1. Data loading and preprocessing
2. Basic data analysis and visualization
3. Model implementation and training
4. Cross-task transfer learning evaluation

Based on the official challenge code with enhancements for analysis and transfer learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import copy
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EEGDataAnalyzer:
    """Comprehensive data analysis for EEG challenge data"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.metadata = None
        self.analysis_results = {}
    
    def load_metadata(self):
        """Load and prepare metadata for analysis"""
        self.metadata = self.dataset.get_metadata()
        print(f"Loaded metadata with {len(self.metadata)} samples")
        return self.metadata
    
    def analyze_response_times(self):
        """Analyze response time distribution and characteristics"""
        if self.metadata is None:
            self.load_metadata()
        
        rt_data = self.metadata['target'].dropna()
        
        print("\n" + "="*50)
        print("RESPONSE TIME ANALYSIS")
        print("="*50)
        
        # Basic statistics
        stats = {
            'count': len(rt_data),
            'mean': rt_data.mean(),
            'std': rt_data.std(),
            'min': rt_data.min(),
            'max': rt_data.max(),
            'median': rt_data.median(),
            'q25': rt_data.quantile(0.25),
            'q75': rt_data.quantile(0.75)
        }
        
        print("Response Time Statistics:")
        for key, value in stats.items():
            print(f"  {key.capitalize()}: {value:.4f}s")
        
        # Distribution analysis
        self.analysis_results['response_times'] = stats
        
        return rt_data
    
    def analyze_by_subject(self):
        """Analyze response times by subject"""
        if self.metadata is None:
            self.load_metadata()
        
        subject_stats = self.metadata.groupby('subject')['target'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print("\n" + "="*50)
        print("SUBJECT-LEVEL ANALYSIS")
        print("="*50)
        print(f"Number of subjects: {len(subject_stats)}")
        print(f"Mean trials per subject: {subject_stats['count'].mean():.1f}")
        print(f"Response time variability across subjects:")
        print(f"  Mean RT range: {subject_stats['mean'].min():.3f}s - {subject_stats['mean'].max():.3f}s")
        print(f"  Mean RT std: {subject_stats['mean'].std():.3f}s")
        
        self.analysis_results['subject_stats'] = subject_stats
        return subject_stats
    
    def analyze_accuracy(self):
        """Analyze accuracy if available"""
        if self.metadata is None:
            self.load_metadata()
        
        if 'correct' in self.metadata.columns:
            accuracy = self.metadata['correct'].mean()
            print(f"\nOverall Accuracy: {accuracy:.3f}")
            
            # Accuracy by subject
            subject_accuracy = self.metadata.groupby('subject')['correct'].mean()
            print(f"Accuracy range across subjects: {subject_accuracy.min():.3f} - {subject_accuracy.max():.3f}")
            
            self.analysis_results['accuracy'] = {
                'overall': accuracy,
                'by_subject': subject_accuracy
            }
        else:
            print("Accuracy information not available in metadata")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.metadata is None:
            self.load_metadata()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EEG Challenge 1: Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Response time distribution
        rt_data = self.metadata['target'].dropna()
        axes[0, 0].hist(rt_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(rt_data.mean(), color='red', linestyle='--', label=f'Mean: {rt_data.mean():.3f}s')
        axes[0, 0].axvline(rt_data.median(), color='green', linestyle='--', label=f'Median: {rt_data.median():.3f}s')
        axes[0, 0].set_xlabel('Response Time (s)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Response Time Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Response time by subject
        subject_means = self.metadata.groupby('subject')['target'].mean()
        axes[0, 1].bar(range(len(subject_means)), subject_means.values, alpha=0.7, color='lightcoral')
        axes[0, 1].set_xlabel('Subject Index')
        axes[0, 1].set_ylabel('Mean Response Time (s)')
        axes[0, 1].set_title('Mean Response Time by Subject')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot of response times
        axes[0, 2].boxplot([rt_data], patch_artist=True, 
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[0, 2].set_ylabel('Response Time (s)')
        axes[0, 2].set_title('Response Time Box Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Response time vs trial number (if available)
        if 'trial' in self.metadata.columns:
            trial_data = self.metadata[['trial', 'target']].dropna()
            axes[1, 0].scatter(trial_data['trial'], trial_data['target'], alpha=0.6, s=20)
            axes[1, 0].set_xlabel('Trial Number')
            axes[1, 0].set_ylabel('Response Time (s)')
            axes[1, 0].set_title('Response Time vs Trial Number')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Trial information\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Trial Information')
        
        # 5. Accuracy analysis (if available)
        if 'correct' in self.metadata.columns:
            accuracy_by_subject = self.metadata.groupby('subject')['correct'].mean()
            axes[1, 1].bar(range(len(accuracy_by_subject)), accuracy_by_subject.values, 
                          alpha=0.7, color='gold')
            axes[1, 1].set_xlabel('Subject Index')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Accuracy by Subject')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Accuracy information\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Accuracy Analysis')
        
        # 6. Response time correlation with other variables
        numeric_cols = self.metadata.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.metadata[numeric_cols].corr()
            im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45)
            axes[1, 2].set_yticklabels(corr_matrix.columns)
            axes[1, 2].set_title('Correlation Matrix')
            plt.colorbar(im, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient numeric\nvariables for correlation', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Correlation Analysis')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*60)
        
        # Load and analyze data
        self.load_metadata()
        rt_data = self.analyze_response_times()
        subject_stats = self.analyze_by_subject()
        self.analyze_accuracy()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)


class EEGModelTrainer:
    """Enhanced model trainer with cross-task transfer learning capabilities"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.training_history = {
            'train_loss': [], 'train_rmse': [],
            'val_loss': [], 'val_rmse': []
        }
        self.best_model_state = None
        self.best_val_rmse = float('inf')
    
    def train_one_epoch(self, dataloader, loss_fn, optimizer, scheduler=None, epoch=0):
        """Train model for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        sum_sq_err = 0.0
        n_samples = 0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                          desc=f"Epoch {epoch}")
        
        for batch_idx, batch in progress_bar:
            X, y = batch[0], batch[1]
            X, y = X.to(self.device).float(), y.to(self.device).float()
            
            optimizer.zero_grad(set_to_none=True)
            preds = self.model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate RMSE
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()
            
            # Update progress bar
            running_rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'RMSE': f"{running_rmse:.6f}"
            })
        
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        rmse_score = (sum_sq_err / max(n_samples, 1)) ** 0.5
        
        return avg_loss, rmse_score
    
    @torch.no_grad()
    def validate(self, dataloader, loss_fn):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        sum_sq_err = 0.0
        n_samples = 0
        
        for batch in tqdm(dataloader, desc="Validation"):
            X, y = batch[0], batch[1]
            X, y = X.to(self.device).float(), y.to(self.device).float()
            
            preds = self.model(X)
            batch_loss = loss_fn(preds, y).item()
            total_loss += batch_loss
            
            preds_flat = preds.detach().view(-1)
            y_flat = y.detach().view(-1)
            sum_sq_err += torch.sum((preds_flat - y_flat) ** 2).item()
            n_samples += y_flat.numel()
        
        avg_loss = total_loss / len(dataloader)
        rmse_score = (sum_sq_err / max(n_samples, 1)) ** 0.5
        
        return avg_loss, rmse_score
    
    def train(self, train_loader, val_loader, n_epochs=100, lr=1e-3, 
              weight_decay=1e-5, patience=10, min_delta=1e-4):
        """Train model with early stopping"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-1)
        loss_fn = nn.MSELoss()
        
        epochs_no_improve = 0
        
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            # Training
            train_loss, train_rmse = self.train_one_epoch(
                train_loader, loss_fn, optimizer, scheduler, epoch
            )
            
            # Validation
            val_loss, val_rmse = self.validate(val_loader, loss_fn)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_rmse'].append(train_rmse)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_rmse'].append(val_rmse)
            
            print(f"Train Loss: {train_loss:.6f}, Train RMSE: {train_rmse:.6f}")
            print(f"Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.6f}")
            
            # Early stopping
            if val_rmse < self.best_val_rmse - min_delta:
                self.best_val_rmse = val_rmse
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                print(f"New best validation RMSE: {val_rmse:.6f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation RMSE: {self.best_val_rmse:.6f}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(self.training_history['val_loss'], label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE plot
        ax2.plot(self.training_history['train_rmse'], label='Train RMSE', color='blue')
        ax2.plot(self.training_history['val_rmse'], label='Val RMSE', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Training and Validation RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @torch.no_grad()
    def evaluate(self, test_loader, loss_fn):
        """Evaluate model on test set"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for batch in tqdm(test_loader, desc="Testing"):
            X, y = batch[0], batch[1]
            X, y = X.to(self.device).float(), y.to(self.device).float()
            
            preds = self.model(X)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        rmse_score = rmse(all_targets, all_preds)
        mae = np.mean(np.abs(all_targets - all_preds))
        r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
        
        # Normalized RMSE (as per challenge requirements)
        normalized_rmse = rmse_score / np.std(all_targets)
        
        print(f"\nTest Results:")
        print(f"RMSE: {rmse_score:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Normalized RMSE: {normalized_rmse:.6f}")
        
        return {
            'rmse': rmse_score,
            'mae': mae,
            'r2': r2,
            'normalized_rmse': normalized_rmse,
            'predictions': all_preds,
            'targets': all_targets
        }


class CrossTaskTransferLearning:
    """Cross-task transfer learning implementation"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.source_models = {}
        self.transfer_results = {}
    
    def train_source_model(self, task_name, train_loader, val_loader, model_class, **model_kwargs):
        """Train a model on source task"""
        print(f"\nTraining source model for task: {task_name}")
        
        model = model_class(**model_kwargs).to(self.device)
        trainer = EEGModelTrainer(model, self.device)
        trainer.train(train_loader, val_loader)
        
        self.source_models[task_name] = {
            'model': model,
            'trainer': trainer
        }
        
        return model, trainer
    
    def transfer_learning(self, source_task, target_task, target_train_loader, 
                         target_val_loader, fine_tune_epochs=20, freeze_features=True):
        """Perform transfer learning from source to target task"""
        print(f"\nTransfer learning: {source_task} -> {target_task}")
        
        if source_task not in self.source_models:
            raise ValueError(f"Source model for {source_task} not found")
        
        # Get source model
        source_model = self.source_models[source_task]['model']
        
        # Create new model with same architecture
        target_model = type(source_model)(**{
            'n_chans': source_model.n_chans,
            'n_outputs': source_model.n_outputs,
            'n_times': source_model.n_times,
            'sfreq': source_model.sfreq
        }).to(self.device)
        
        # Copy weights from source model
        target_model.load_state_dict(source_model.state_dict())
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for name, param in target_model.named_parameters():
                if 'classifier' not in name and 'fc' not in name:
                    param.requires_grad = False
            print("Frozen feature extraction layers")
        
        # Fine-tune on target task
        trainer = EEGModelTrainer(target_model, self.device)
        trainer.train(target_train_loader, target_val_loader, n_epochs=fine_tune_epochs)
        
        # Store results
        transfer_key = f"{source_task}_to_{target_task}"
        self.transfer_results[transfer_key] = {
            'model': target_model,
            'trainer': trainer
        }
        
        return target_model, trainer


def main():
    """Main execution function"""
    print("EEG Challenge 1: Cross-Task Transfer Learning Implementation")
    print("="*70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Note: This is a template implementation
    # In practice, you would need to:
    # 1. Install eegdash and braindecode
    # 2. Download the actual dataset
    # 3. Run the complete pipeline
    
    print("\nTo run the complete implementation:")
    print("1. Install required packages: pip install eegdash braindecode")
    print("2. Download the dataset using eegdash")
    print("3. Run the data analysis and model training")
    print("4. Implement cross-task transfer learning")
    
    print("\nThis implementation provides:")
    print("- Comprehensive data analysis tools")
    print("- Enhanced model training with early stopping")
    print("- Cross-task transfer learning framework")
    print("- Evaluation metrics including normalized RMSE")
    print("- Visualization tools for analysis and training progress")


if __name__ == "__main__":
    main()

