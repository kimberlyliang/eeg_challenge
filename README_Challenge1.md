# Challenge 1: Cross-Task Transfer Learning for Response Time Prediction

This repository contains a comprehensive framework for Challenge 1 of the EEG challenge, which focuses on cross-task transfer learning for response time prediction.

## Overview

Challenge 1 requires predicting response times across different EEG tasks using cross-task transfer learning. The evaluation metric is the normalized root mean square error (RMSE):

```
score = rmse(y_true, y_pred) / std(y_true)
```

## Dataset Structure

The dataset contains both **active tasks** (with response times) and **passive tasks** (without response times):

### Active Tasks (Response Time Prediction Targets)
1. **Contrast Change Detection** - Button press responses to visual targets
2. **Sequence Learning (6-target)** - Mouse click responses for sequence reproduction
3. **Sequence Learning (8-target)** - Mouse click responses for sequence reproduction  
4. **Symbol Search** - Yes/No responses for symbol matching

### Passive Tasks (No Response Times)
1. **Resting State** - Eyes open/closed periods
2. **Surround Suppression** - Visual perception task
3. **Movie Watching** - Passive viewing of videos (Despicable Me, Diary of a Wimpy Kid, Fun with Fractals, The Present)

## Files

- `challenge1_framework.py` - Main framework class with comprehensive functionality
- `response_time_analysis.py` - Focused analysis script for response time extraction
- `demo_challenge1.py` - Demonstration script showing how to use the framework
- `model.py` - Your existing model file (can be extended)

## Quick Start

### 1. Run the Demo
```bash
python demo_challenge1.py
```

This will demonstrate:
- Response time extraction from individual tasks
- Cross-task analysis across multiple participants
- The normalized RMSE evaluation metric

### 2. Extract Response Times
```python
from response_time_analysis import ResponseTimeAnalyzer

# Initialize analyzer
analyzer = ResponseTimeAnalyzer("/path/to/local_directory")

# Extract response times for specific participants
participants = ["sub-NDARAC904DMU", "sub-NDARAM704GKZ"]
response_times_df = analyzer.get_all_response_times(participants)

# Analyze the data
analyzer.analyze_response_times(response_times_df)
```

### 3. Implement Cross-Task Transfer Learning
```python
# Create features for transfer learning
X, y, task_labels = analyzer.create_cross_task_features(response_times_df)

# Evaluate cross-task transfer
results = analyzer.evaluate_cross_task_transfer(X, y, task_labels)
```

## Response Time Extraction Details

### Contrast Change Detection
- **Target Events**: `right_target`, `left_target`
- **Response Events**: `right_buttonPress`, `left_buttonPress`
- **Response Time**: Time between target presentation and button press
- **Trials per participant**: ~36 (12 per run × 3 runs)

### Sequence Learning
- **Learning Blocks**: `learningBlock_1` through `learningBlock_5`
- **Response Time**: Time from block start to user response
- **Data**: User answers and correct answers stored in event files
- **Trials per participant**: 5 (one per learning block)

### Symbol Search
- **Response Events**: `trialResponse`
- **Response Time**: Time from task start to response
- **Data**: User answers and correct answers
- **Trials per participant**: Variable (up to 60 trials in 2 minutes)

## Evaluation Metric

The normalized RMSE is calculated as:

```python
from sklearn.metrics import root_mean_squared_error as rmse
import numpy as np

def calculate_normalized_rmse(y_true, y_pred):
    rmse_score = rmse(y_true, y_pred)
    std_true = np.std(y_true)
    return rmse_score / std_true
```

**Interpretation:**
- Lower values indicate better performance
- Values close to 0 indicate excellent predictions
- Values around 1 indicate predictions are as good as using the mean
- Values > 1 indicate worse than using the mean

## Next Steps for Implementation

1. **Extract EEG Features**: Use the `.bdf` files to extract meaningful EEG features
2. **Feature Engineering**: Create task-specific and cross-task features
3. **Model Development**: Implement transfer learning models (e.g., domain adaptation, multi-task learning)
4. **Cross-Validation**: Use proper cross-validation strategies for transfer learning
5. **Hyperparameter Tuning**: Optimize model parameters for each task transfer

## Example Transfer Learning Scenarios

1. **Contrast Change → Symbol Search**: Transfer visual attention patterns
2. **Sequence Learning → Contrast Change**: Transfer cognitive processing patterns
3. **Multi-task Learning**: Train on all tasks simultaneously
4. **Domain Adaptation**: Adapt features from one task to another

## Data Quality Considerations

- Some participants may have missing data for certain tasks
- Response times may vary significantly across participants
- Task difficulty and participant age may affect response times
- Consider participant-specific normalization

## Troubleshooting

1. **No response times found**: Check that participant has data for the specific task
2. **Empty dataframes**: Verify file paths and participant IDs
3. **Import errors**: Ensure all required packages are installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)

## Contributing

Feel free to extend the framework with:
- Additional feature extraction methods
- More sophisticated transfer learning algorithms
- Better visualization tools
- Performance optimization

## License

This framework is provided as-is for the EEG challenge. Please cite appropriately if used in research.
