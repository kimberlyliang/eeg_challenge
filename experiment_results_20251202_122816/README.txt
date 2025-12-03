======================================================================
EXPERIMENT RESULTS DIRECTORY
======================================================================

Created: 2025-12-02 12:28:16
Directory: experiment_results_20251202_122816

This directory contains all results from the training experiments.

STRUCTURE:
  - <model_name>/          : Individual model directories
    - best_model.pt       : Best model weights
    - best_checkpoint.pt  : Full checkpoint with optimizer state
    - checkpoint_epoch_*.pt: Periodic checkpoints
    - training_history.json: Complete training history
    - training_curves.png : Training/validation curves
    - config.json         : Model configuration
    - final_*.npy        : Final predictions, targets, errors
  - *_best.pt             : Quick access to best models
  - *_error_distribution.png: Error distribution plots
  - *_error_stats.json   : Error statistics
  - *_demographics.png   : Demographic analysis plots
  - *_demographics.json  : Demographic statistics
  - ablation_studies.json: Ablation study framework
  - experiment_summary.json: Complete summary (JSON)
  - EXPERIMENT_SUMMARY.txt: Human-readable summary
