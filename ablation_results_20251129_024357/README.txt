======================================================================
ABLATION STUDIES RESULTS
======================================================================

Created: 2025-11-30 09:22:26
Directory: ablation_results_20251129_024357

This directory contains all results from the ablation studies.

STRUCTURE:
  - <experiment_name>/     : Individual experiment directories
    - best_model.pt        : Best model weights
    - checkpoint_epoch_*.pt: Periodic checkpoints
    - training_history.json: Complete training history
    - training_curves.png  : Training/validation curves
  - *_best.pt              : Quick access to best models
  - ablation_results.json  : Complete results (JSON)
  - ablation_summary.json  : Summary with best configurations
  - ABLATION_SUMMARY.txt   : Human-readable summary
  - ablation_comparison.png: Comparison plots
  - <study_name>_curves.png: Learning curves per study

KEY FILES TO REVIEW:
  1. ABLATION_SUMMARY.txt - Start here for overview
  2. ablation_comparison.png - Visual comparison
  3. ablation_summary.json - Best configurations
