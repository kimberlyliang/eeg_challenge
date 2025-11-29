# Ablation Studies Script Guide

## Overview
The `run_ablation_studies.py` script runs comprehensive ablation experiments overnight to test different hyperparameters and model configurations.

## What It Tests

### 1. **Learning Rate** (4 experiments)
- Values: 5e-4, 1e-3, 2e-3, 5e-3
- Model: CNN-only baseline
- Tests optimal learning rate for convergence

### 2. **Batch Size** (3 experiments)
- Values: 32, 64, 128
- Model: CNN-only baseline
- Tests impact of batch size on training stability

### 3. **Dropout Rate** (4 experiments)
- Values: 0.0, 0.1, 0.3, 0.5
- Model: CNN + Transformer
- Tests regularization strength

### 4. **Transformer Layers** (4 experiments)
- Values: 1, 2, 3, 4 layers
- Model: CNN + Transformer
- Tests optimal transformer depth

### 5. **Transformer Attention Heads** (3 experiments)
- Values: 2, 4, 8 heads
- Model: CNN + Transformer
- Tests optimal attention mechanism size

### 6. **Transformer Dimension (d_model)** (3 experiments)
- Values: 32, 64, 128
- Model: CNN + Transformer
- Tests optimal feature dimension

**Total: ~21 experiments × 20 epochs = ~420 epochs total**

## What Gets Saved

### For Each Experiment:
- `best_model.pt` - Best model weights
- `checkpoint_epoch_*.pt` - Checkpoints every 5 epochs
- `training_history.json` - Complete training history
- `training_curves.png` - Training/validation curves

### Summary Files:
- `ablation_results.json` - Complete results (JSON)
- `ablation_summary.json` - Best configurations summary
- `ABLATION_SUMMARY.txt` - Human-readable summary
- `ablation_comparison.png` - Comparison bar charts
- `<study_name>_curves.png` - Learning curves per study
- `README.txt` - Directory guide

## How to Run

```bash
# Activate environment
source eeg_env/bin/activate

# Run overnight
nohup python run_ablation_studies.py > ablation_run.log 2>&1 &

# Or run in screen/tmux session
screen -S ablation
python run_ablation_studies.py
# Press Ctrl+A then D to detach
```

## Expected Runtime

- ~20 epochs per experiment
- ~21 experiments total
- Estimated: 4-8 hours (depending on GPU/CPU)

## Results Location

All results saved to: `ablation_results_YYYYMMDD_HHMMSS/`

## What to Check in the Morning

1. **ABLATION_SUMMARY.txt** - Start here for overview
2. **ablation_comparison.png** - Visual comparison of all experiments
3. **ablation_summary.json** - Best configurations
4. Individual experiment folders for detailed results

## Configuration

You can enable/disable studies by editing `ABLATION_CONFIG` at the top of the script:

```python
ABLATION_CONFIG = {
    'learning_rate': {'enabled': True, ...},
    'batch_size': {'enabled': True, ...},
    # etc.
}
```

## Notes

- Each experiment runs for 20 epochs (reduced from 30 for speed)
- Best model is saved based on validation RMSE
- All experiments use the same train/val/test split
- Results are saved incrementally (won't lose progress if interrupted)

