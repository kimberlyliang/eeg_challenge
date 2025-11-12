"""
Demo script for Challenge 1: Cross-Task Transfer Learning

This script demonstrates how to:
1. Extract response times from EEG tasks
2. Implement cross-task transfer learning
3. Evaluate using the normalized RMSE metric

Run this script to see the framework in action!
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from response_time_analysis import ResponseTimeAnalyzer
import pandas as pd
import numpy as np

def demo_response_time_extraction():
    """Demonstrate response time extraction from a single participant"""
    print("=" * 60)
    print("DEMO: Response Time Extraction")
    print("=" * 60)
    
    # Initialize analyzer
    data_root = "/Users/kimberly/Documents/ESE5380/eeg_challenge/local_directory"
    analyzer = ResponseTimeAnalyzer(data_root)
    
    # Get first participant
    participant_id = analyzer.participants_df['participant_id'].iloc[0]
    print(f"Analyzing participant: {participant_id}")
    
    # Extract response times for each task
    print("\n1. Contrast Change Detection:")
    contrast_trials = analyzer.extract_contrast_change_response_times(participant_id)
    if not contrast_trials.empty:
        print(f"   Found {len(contrast_trials)} trials")
        print(f"   Mean response time: {contrast_trials['response_time'].mean():.3f}s")
        print(f"   Accuracy: {contrast_trials['correct'].mean():.3f}")
    else:
        print("   No trials found")
    
    print("\n2. Sequence Learning:")
    seq_trials = analyzer.extract_sequence_learning_response_times(participant_id)
    if not seq_trials.empty:
        print(f"   Found {len(seq_trials)} trials")
        print(f"   Mean response time: {seq_trials['response_time'].mean():.3f}s")
        print(f"   Accuracy: {seq_trials['correct'].mean():.3f}")
    else:
        print("   No trials found")
    
    print("\n3. Symbol Search:")
    symbol_trials = analyzer.extract_symbol_search_response_times(participant_id)
    if not symbol_trials.empty:
        print(f"   Found {len(symbol_trials)} trials")
        print(f"   Mean response time: {symbol_trials['response_time'].mean():.3f}s")
        print(f"   Accuracy: {symbol_trials['correct'].mean():.3f}")
    else:
        print("   No trials found")

def demo_cross_task_analysis():
    """Demonstrate cross-task analysis with multiple participants"""
    print("\n" + "=" * 60)
    print("DEMO: Cross-Task Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    data_root = "/Users/kimberly/Documents/ESE5380/eeg_challenge/local_directory"
    analyzer = ResponseTimeAnalyzer(data_root)
    
    # Process first 3 participants
    sample_participants = analyzer.participants_df['participant_id'].head(3).tolist()
    print(f"Processing participants: {sample_participants}")
    
    # Extract all response times
    print("\nExtracting response times...")
    response_times_df = analyzer.get_all_response_times(sample_participants)
    
    if response_times_df.empty:
        print("No response time data found!")
        return
    
    print(f"\nExtracted {len(response_times_df)} total trials")
    print(f"Tasks: {response_times_df['task'].unique()}")
    print(f"Participants: {response_times_df['participant_id'].nunique()}")
    
    # Show summary statistics
    print("\nSummary Statistics by Task:")
    summary = response_times_df.groupby('task')['response_time'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(summary)
    
    # Create features and evaluate transfer learning
    print("\nCreating features for transfer learning...")
    X, y, task_labels = analyzer.create_cross_task_features(response_times_df)
    
    if len(X) > 0:
        print(f"Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Evaluate cross-task transfer
        print("\nEvaluating cross-task transfer learning...")
        results = analyzer.evaluate_cross_task_transfer(X, y, task_labels)
        
        if results:
            print("\nBest transfer learning results:")
            best_transfer = min(results.items(), key=lambda x: x[1]['normalized_rmse'])
            print(f"  {best_transfer[0]}: Normalized RMSE = {best_transfer[1]['normalized_rmse']:.4f}")

def demo_evaluation_metric():
    """Demonstrate the normalized RMSE evaluation metric"""
    print("\n" + "=" * 60)
    print("DEMO: Normalized RMSE Evaluation Metric")
    print("=" * 60)
    
    from sklearn.metrics import root_mean_squared_error as rmse
    
    # Example 1: Perfect predictions
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rmse_score = rmse(y_true, y_pred)
    std_true = np.std(y_true)
    normalized_rmse = rmse_score / std_true
    print(f"Perfect predictions:")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  Std of true values: {std_true:.4f}")
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    
    # Example 2: Good predictions
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    rmse_score = rmse(y_true, y_pred)
    normalized_rmse = rmse_score / std_true
    print(f"\nGood predictions:")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    
    # Example 3: Poor predictions
    y_pred = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    rmse_score = rmse(y_true, y_pred)
    normalized_rmse = rmse_score / std_true
    print(f"\nPoor predictions:")
    print(f"  RMSE: {rmse_score:.4f}")
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    
    print(f"\nNote: Lower normalized RMSE values indicate better performance")

def main():
    """Run all demonstrations"""
    print("EEG Challenge 1: Cross-Task Transfer Learning Demo")
    print("This demo shows how to extract response times and implement transfer learning")
    
    try:
        # Demo 1: Response time extraction
        demo_response_time_extraction()
        
        # Demo 2: Cross-task analysis
        demo_cross_task_analysis()
        
        # Demo 3: Evaluation metric
        demo_evaluation_metric()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Extract EEG features from the .bdf files")
        print("2. Implement more sophisticated transfer learning models")
        print("3. Use the normalized RMSE metric for evaluation")
        print("4. Experiment with different feature representations")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure the data path is correct and files exist")

if __name__ == "__main__":
    main()

