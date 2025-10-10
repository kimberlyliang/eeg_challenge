import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class ResponseTimeAnalyzer:
    """Analyzer for extracting and analyzing response times from EEG tasks"""
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.participants_df = pd.read_csv(self.data_root / "participants.tsv", sep='\t')
        print(f"Loaded {len(self.participants_df)} participants")
    
    def extract_contrast_change_response_times(self, participant_id: str) -> pd.DataFrame:
        """Extract response times from contrast change detection task"""
        trials = []
        
        for run in [1, 2, 3]:
            event_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-contrastChangeDetection_run-{run}_events.tsv"
            
            if not event_file.exists():
                continue
                
            events_df = pd.read_csv(event_file, sep='\t')
            targets = events_df[events_df['value'].isin(['right_target', 'left_target'])]
            responses = events_df[events_df['value'].isin(['right_buttonPress', 'left_buttonPress'])]
            
            for _, target in targets.iterrows():
                target_time = target['onset']
                target_type = target['value']
                
                response_window = responses[
                    (responses['onset'] > target_time) & 
                    (responses['onset'] < target_time + 5.0)
                ]
                
                if len(response_window) > 0:
                    response = response_window.iloc[0]
                    response_time = response['onset'] - target_time
                    
                    trials.append({
                        'participant_id': participant_id,
                        'task': 'contrastChangeDetection',
                        'run': run,
                        'trial_id': f"run{run}_trial{len(trials)}",
                        'target_type': target_type,
                        'response_type': response['value'],
                        'response_time': response_time,
                        'correct': self._is_correct_response(target_type, response['value']),
                        'target_onset': target_time,
                        'response_onset': response['onset']
                    })
        
        return pd.DataFrame(trials)
    
    def extract_sequence_learning_response_times(self, participant_id: str) -> pd.DataFrame:
        """Extract response times from sequence learning task"""
        trials = []
        
        # Check both 6-target and 8-target versions
        for target_count in [6, 8]:
            event_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-seqLearning{target_count}target_events.tsv"
            
            if not event_file.exists():
                continue
                
            events_df = pd.read_csv(event_file, sep='\t')
            
            # Find learning blocks
            learning_blocks = events_df[events_df['value'].str.contains('learningBlock_', na=False)]
            
            for _, block in learning_blocks.iterrows():
                block_num = block['value'].split('_')[1]
                block_start = block['onset']
                
                # response for block 
                response_row = events_df[
                    (events_df['onset'] > block_start) & 
                    (events_df['user_answer'].notna())
                ]
                
                if len(response_row) > 0:
                    response = response_row.iloc[0]
                    response_time = response['onset'] - block_start
                    
                    trials.append({
                        'participant_id': participant_id,
                        'task': f'seqLearning{target_count}target',
                        'run': 1,  # Only one run per target count
                        'trial_id': f"block_{block_num}",
                        'block_number': int(block_num),
                        'response_time': response_time,
                        'user_answer': response['user_answer'],
                        'correct_answer': response['correct_answer'],
                        'correct': response['user_answer'] == response['correct_answer'],
                        'block_start': block_start,
                        'response_onset': response['onset']
                    })
        
        return pd.DataFrame(trials)
    
    def extract_symbol_search_response_times(self, participant_id: str) -> pd.DataFrame:
        """Extract response times from symbol search task"""
        trials = []
        
        event_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-symbolSearch_events.tsv"
        
        if not event_file.exists():
            return pd.DataFrame()
            
        events_df = pd.read_csv(event_file, sep='\t')
        
        # Find trial responses
        responses = events_df[events_df['value'] == 'trialResponse']
        
        for i, (_, response) in enumerate(responses.iterrows()):
            trials.append({
                'participant_id': participant_id,
                'task': 'symbolSearch',
                'run': 1,
                'trial_id': f"trial_{i+1}",
                'response_time': response['onset'],  # Time from task start
                'user_answer': response['user_answer'],
                'correct_answer': response['correct_answer'],
                'correct': response['user_answer'] == response['correct_answer'],
                'response_onset': response['onset']
            })
        
        return pd.DataFrame(trials)
    
    def _is_correct_response(self, target_type: str, response_type: str) -> bool:
        """Check if response matches target type"""
        if target_type == 'right_target' and response_type == 'right_buttonPress':
            return True
        elif target_type == 'left_target' and response_type == 'left_buttonPress':
            return True
        return False
    
    def get_all_response_times(self, participant_ids: list = None) -> pd.DataFrame:
        """Extract response times for all specified participants"""
        if participant_ids is None:
            participant_ids = self.participants_df['participant_id'].tolist()
        
        all_trials = []
        
        for participant_id in participant_ids:
            print(f"Processing {participant_id}...")
            
            # Extract from each task type
            contrast_trials = self.extract_contrast_change_response_times(participant_id)
            seq_trials = self.extract_sequence_learning_response_times(participant_id)
            symbol_trials = self.extract_symbol_search_response_times(participant_id)
            
            all_trials.extend([
                contrast_trials,
                seq_trials,
                symbol_trials
            ])
        
        # Combine all trials
        combined_df = pd.concat([df for df in all_trials if not df.empty], ignore_index=True)
        return combined_df
    
    def analyze_response_times(self, df: pd.DataFrame):
        """Analyze and visualize response time data"""
        if df.empty:
            print("No response time data found")
            return
        
        print(f"\nTotal trials: {len(df)}")
        print(f"Participants: {df['participant_id'].nunique()}")
        print(f"Tasks: {df['task'].unique()}")
        
        # Basic statistics
        print("\nResponse Time Statistics by Task:")
        print(df.groupby('task')['response_time'].describe())
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Distribution by task
        plt.subplot(2, 3, 1)
        for task in df['task'].unique():
            task_data = df[df['task'] == task]['response_time']
            plt.hist(task_data, alpha=0.7, label=task, bins=20)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distributions')
        plt.legend()
        
        # Box plot by task
        plt.subplot(2, 3, 2)
        df.boxplot(column='response_time', by='task', ax=plt.gca())
        plt.title('Response Time Box Plots')
        plt.suptitle('')
        
        # Accuracy by task (if available)
        if 'correct' in df.columns:
            plt.subplot(2, 3, 3)
            accuracy = df.groupby('task')['correct'].mean()
            accuracy.plot(kind='bar', ax=plt.gca())
            plt.title('Accuracy by Task')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
        
        # Response time vs trial number
        plt.subplot(2, 3, 4)
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            if 'trial_id' in task_data.columns:
                trial_nums = range(len(task_data))
                plt.scatter(trial_nums, task_data['response_time'], alpha=0.6, label=task)
        plt.xlabel('Trial Number')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time vs Trial Number')
        plt.legend()
        
        # Participant-level analysis
        plt.subplot(2, 3, 5)
        participant_means = df.groupby('participant_id')['response_time'].mean()
        plt.hist(participant_means, bins=15)
        plt.xlabel('Mean Response Time (seconds)')
        plt.ylabel('Number of Participants')
        plt.title('Participant Mean Response Times')
        
        # Task comparison
        plt.subplot(2, 3, 6)
        task_means = df.groupby('task')['response_time'].mean()
        task_means.plot(kind='bar', ax=plt.gca())
        plt.title('Mean Response Times by Task')
        plt.ylabel('Mean Response Time (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_cross_task_features(self, df: pd.DataFrame) -> tuple:
        """Create features for cross-task transfer learning"""
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Basic features (in practice, you would extract EEG features here)
        features = []
        targets = []
        task_labels = []
        
        for _, trial in df.iterrows():
            # Placeholder features - in practice, extract from EEG data
            feature_vector = [
                trial['response_time'],  # Current response time as feature
                len(trial['trial_id']),  # Trial ID length (placeholder)
                1 if trial['task'] == 'contrastChangeDetection' else 0,  # Task indicator
                1 if trial['task'].startswith('seqLearning') else 0,
                1 if trial['task'] == 'symbolSearch' else 0,
                trial.get('correct', 0.5),  # Accuracy (if available)
                trial.get('block_number', 0),  # Block number (if available)
                trial['run'],  # Run number
                hash(trial['participant_id']) % 100,  # Participant hash (placeholder)
                np.random.random()  # Random feature (placeholder)
            ]
            
            features.append(feature_vector)
            targets.append(trial['response_time'])
            task_labels.append(trial['task'])
        
        return np.array(features), np.array(targets), task_labels
    
    def evaluate_cross_task_transfer(self, X: np.ndarray, y: np.ndarray, task_labels: list):
        """Evaluate cross-task transfer learning performance"""
        if len(X) == 0:
            print("No data for evaluation")
            return
        
        # Split data by task
        unique_tasks = list(set(task_labels))
        print(f"Evaluating transfer across {len(unique_tasks)} tasks: {unique_tasks}")
        
        results = {}
        
        for source_task in unique_tasks:
            for target_task in unique_tasks:
                if source_task == target_task:
                    continue
                
                # Get source and target data
                source_mask = np.array(task_labels) == source_task
                target_mask = np.array(task_labels) == target_task
                
                if not np.any(source_mask) or not np.any(target_mask):
                    continue
                
                X_source = X[source_mask]
                y_source = y[source_mask]
                X_target = X[target_mask]
                y_target = y[target_mask]
                
                # Train on source task
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_source, y_source)
                
                # Test on target task
                y_pred = model.predict(X_target)
                
                # Calculate normalized RMSE
                rmse_score = rmse(y_target, y_pred)
                std_target = np.std(y_target)
                normalized_rmse = rmse_score / std_target if std_target > 0 else float('inf')
                
                results[f"{source_task} -> {target_task}"] = {
                    'rmse': rmse_score,
                    'normalized_rmse': normalized_rmse,
                    'n_source': len(X_source),
                    'n_target': len(X_target)
                }
        
        # Print results
        print("\nCross-Task Transfer Learning Results:")
        print("=" * 60)
        for transfer, metrics in results.items():
            print(f"{transfer:30} | RMSE: {metrics['rmse']:.4f} | Norm RMSE: {metrics['normalized_rmse']:.4f}")
        
        return results


def main():
    """Main analysis function"""
    print("EEG Challenge 1: Cross-Task Transfer Learning Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    data_root = "/Users/kimberly/Documents/ESE5380/eeg_challenge/local_directory"
    analyzer = ResponseTimeAnalyzer(data_root)
    
    # Process a subset of participants for demonstration
    sample_participants = analyzer.participants_df['participant_id'].head(5).tolist()
    print(f"Processing {len(sample_participants)} participants: {sample_participants}")
    
    # Extract response times
    print("\nExtracting response times...")
    response_times_df = analyzer.get_all_response_times(sample_participants)
    
    if response_times_df.empty:
        print("No response time data found!")
        return
    
    # Analyze response times
    print("\nAnalyzing response times...")
    analyzer.analyze_response_times(response_times_df)
    
    # Create features for transfer learning
    print("\nCreating features for cross-task transfer learning...")
    X, y, task_labels = analyzer.create_cross_task_features(response_times_df)
    
    if len(X) > 0:
        print(f"Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Evaluate cross-task transfer
        print("\nEvaluating cross-task transfer learning...")
        results = analyzer.evaluate_cross_task_transfer(X, y, task_labels)
        
        # Save results
        output_dir = Path("/Users/kimberly/Documents/ESE5380/eeg_challenge/output")
        output_dir.mkdir(exist_ok=True)
        
        response_times_df.to_csv(output_dir / "response_times.csv", index=False)
        print(f"\nSaved response times to {output_dir / 'response_times.csv'}")
        
        if results:
            results_df = pd.DataFrame(results).T
            results_df.to_csv(output_dir / "transfer_learning_results.csv")
            print(f"Saved transfer learning results to {output_dir / 'transfer_learning_results.csv'}")


if __name__ == "__main__":
    main()
