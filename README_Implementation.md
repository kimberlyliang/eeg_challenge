# EEG Challenge 1: Complete Implementation Guide

This repository contains a comprehensive implementation for Challenge 1: Cross-Task Transfer Learning for Response Time Prediction. The implementation includes data analysis, model training, and transfer learning capabilities.

## 📁 Files Overview

### Core Implementation Files
- `eeg_challenge_implementation.py` - Complete framework with data analysis and model training
- `practical_eeg_analysis.py` - Practical implementation that works with your local data
- `demo_models.py` - Simple demonstration script that runs immediately
- `challenge1_framework.py` - Original framework for response time extraction
- `response_time_analysis.py` - Focused analysis script

### Documentation
- `README_Challenge1.md` - Original challenge explanation
- `README_Implementation.md` - This file

## 🚀 Quick Start

### Option 1: Run the Demo (Recommended for first-time users)
```bash
python demo_models.py
```
This will:
- Create synthetic EEG data
- Train multiple models (EEGNeX, SimpleEEGNet, CNN1D)
- Show training curves and evaluation metrics
- Demonstrate the normalized RMSE evaluation metric

### Option 2: Run with Your Local Data
```bash
python practical_eeg_analysis.py
```
This will:
- Load your local EEG dataset
- Extract response times from different tasks
- Perform data analysis and visualization
- Train models on synthetic data (since real EEG data requires special libraries)

### Option 3: Full Framework (Advanced)
```bash
python eeg_challenge_implementation.py
```
This provides the complete framework but requires the full dataset to be downloaded.

## 🧠 Model Implementations

### 1. EEGNeX Model
Based on the official braindecode implementation:
- Temporal convolution for time-series processing
- Spatial convolution for channel relationships
- Multi-layer feature extraction
- Global average pooling
- Dense classifier with dropout

### 2. SimpleEEGNet Model
Simplified version of EEGNet:
- 2D convolutions for temporal and spatial processing
- Batch normalization and ELU activation
- Pooling layers for dimensionality reduction
- Dense classifier

### 3. CNN1D Model
Simple 1D CNN for comparison:
- Multiple 1D convolution layers
- Max pooling for feature extraction
- Dense classifier

## 📊 Data Analysis Features

### Response Time Analysis
- Distribution analysis (mean, std, min, max, quartiles)
- Subject-level analysis
- Task-specific analysis
- Accuracy analysis (when available)

### Visualizations
- Response time distributions
- Box plots by task and subject
- Correlation matrices
- Training curves
- Prediction vs target scatter plots
- Residual analysis

## 🔄 Cross-Task Transfer Learning

The framework supports transfer learning between different EEG tasks:

1. **Source Task Training**: Train a model on one task (e.g., Resting State)
2. **Target Task Fine-tuning**: Fine-tune the model on another task (e.g., Contrast Change Detection)
3. **Evaluation**: Compare performance with and without transfer learning

### Example Transfer Learning Scenarios
- **Passive → Active**: Transfer from Resting State to Contrast Change Detection
- **Visual → Cognitive**: Transfer from Surround Suppression to Symbol Search
- **Memory → Attention**: Transfer from Sequence Learning to Contrast Change Detection

## 📈 Evaluation Metrics

### Primary Metric: Normalized RMSE
```
normalized_rmse = rmse(y_true, y_pred) / std(y_true)
```

### Additional Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Accuracy**: For classification tasks (when available)

## 🛠️ Installation Requirements

### Basic Requirements
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
```

### For Full Dataset (Optional)
```bash
pip install eegdash braindecode
pip install mne mne-bids
```

## 📋 Usage Examples

### 1. Basic Model Training
```python
from demo_models import EEGNeX, train_model, evaluate_model

# Create model
model = EEGNeX(n_chans=129, n_outputs=1, n_times=200)

# Train model
train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)

# Evaluate model
results = evaluate_model(model, X_test, y_test)
print(f"Normalized RMSE: {results['normalized_rmse']:.4f}")
```

### 2. Data Analysis
```python
from practical_eeg_analysis import EEGDataProcessor

# Initialize processor
processor = EEGDataProcessor("/path/to/local_directory")

# Load participants
processor.load_participants()

# Extract response times
rt_data = processor.extract_response_times("sub-NDARAC904DMU", "contrastChangeDetection")

# Analyze data
print(f"Found {len(rt_data)} trials")
print(f"Mean response time: {rt_data['response_time'].mean():.3f}s")
```

### 3. Cross-Task Transfer Learning
```python
from eeg_challenge_implementation import CrossTaskTransferLearning

# Initialize transfer learning
transfer = CrossTaskTransferLearning(device='cuda')

# Train source model
source_model, _ = transfer.train_source_model(
    "resting_state", train_loader, val_loader, EEGNeX
)

# Transfer to target task
target_model, _ = transfer.transfer_learning(
    "resting_state", "contrast_change", 
    target_train_loader, target_val_loader
)
```

## 🎯 Challenge 1 Requirements

### Task Description
Predict response times from EEG data using cross-task transfer learning.

### Data Format
- **Input**: EEG data (129 channels × 200 time points = 2 seconds)
- **Output**: Response time prediction (regression)
- **Tasks**: Contrast Change Detection, Sequence Learning, Symbol Search

### Evaluation
- **Metric**: Normalized RMSE
- **Cross-validation**: Subject-level splits
- **Transfer Learning**: From passive to active tasks

## 🔧 Customization

### Adding New Models
```python
class YourModel(nn.Module):
    def __init__(self, n_chans=129, n_outputs=1, n_times=200):
        super().__init__()
        # Your model architecture
        pass
    
    def forward(self, x):
        # Your forward pass
        return output
```

### Custom Data Loading
```python
def load_custom_data(data_path):
    # Your data loading logic
    return X, y
```

### Custom Evaluation
```python
def custom_evaluation(predictions, targets):
    # Your evaluation metrics
    return metrics
```

## 📚 Key Concepts

### EEG Data Preprocessing
- **Sampling Rate**: 100 Hz
- **Channels**: 129 (including reference)
- **Window Length**: 2 seconds (200 samples)
- **Epoching**: Stimulus-locked with 500ms offset

### Response Time Extraction
- **Contrast Change Detection**: Time from target to button press
- **Sequence Learning**: Time from block start to response
- **Symbol Search**: Time from task start to response

### Transfer Learning Strategy
1. **Pre-training**: Train on source task (e.g., Resting State)
2. **Fine-tuning**: Adapt to target task (e.g., Contrast Change Detection)
3. **Evaluation**: Compare with direct training on target task

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Data not found**: Check file paths and participant IDs
3. **Import errors**: Install required packages
4. **Model convergence**: Adjust learning rate or architecture

### Performance Tips
1. **Use GPU**: Significantly faster training
2. **Batch size**: Start with 32, adjust based on memory
3. **Learning rate**: Start with 1e-3, use learning rate scheduling
4. **Early stopping**: Prevent overfitting

## 📖 References

- [Challenge 1 Description](https://eeg2025.github.io/)
- [Braindecode Documentation](https://braindecode.org/)
- [EEGDash Documentation](https://eeglab.org/EEGDash/)
- [HBN-EEG Dataset](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2)

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving data analysis tools
- Enhancing transfer learning methods
- Optimizing performance
- Adding new evaluation metrics

## 📄 License

This implementation is provided for educational and research purposes. Please cite appropriately if used in research.

---

**Happy coding and good luck with the challenge! 🧠⚡**
