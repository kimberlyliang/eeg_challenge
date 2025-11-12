#!/usr/bin/env python3
"""
Real EEG Feature Extraction for Challenge 1

This script loads your actual EEG data from .bdf files and extracts
comprehensive features including catch22 features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import json
from datetime import datetime
import logging

# MNE imports
import mne
from mne.io import read_raw_bdf
from mne import events_from_annotations
from mne.preprocessing import ICA

# Feature extraction
try:
    from catch22 import catch22_all
    CATCH22_AVAILABLE = True
except ImportError:
    print("catch22 not available. Install with: pip install catch22")
    CATCH22_AVAILABLE = False

try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    from scipy.stats import entropy
    from scipy.signal import welch, coherence
    SCIPY_AVAILABLE = True
except ImportError:
    print("scipy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_eeg_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealEEGFeatureExtractor:
    """Real EEG feature extraction from .bdf files"""
    
    def __init__(self, data_root: str, sfreq=100, n_chans=129, window_length=2.0):
        self.data_root = Path(data_root)
        self.sfreq = sfreq
        self.n_chans = n_chans
        self.window_length = window_length
        self.window_samples = int(window_length * sfreq)
        
        # Frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        logger.info(f"Initialized RealEEGFeatureExtractor")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Window length: {window_length}s ({self.window_samples} samples)")
    
    def load_participant_data(self, participant_id: str, task_name: str, run=None):
        """Load EEG data for a specific participant and task"""
        logger.info(f"Loading data for {participant_id} - {task_name}")
        
        # Construct file path
        if run is not None:
            bdf_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-{task_name}_run-{run}_eeg.bdf"
            events_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-{task_name}_run-{run}_events.tsv"
        else:
            bdf_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-{task_name}_eeg.bdf"
            events_file = self.data_root / f"{participant_id}/eeg/{participant_id}_task-{task_name}_events.tsv"
        
        if not bdf_file.exists():
            logger.warning(f"BDF file not found: {bdf_file}")
            return None, None, None
        
        if not events_file.exists():
            logger.warning(f"Events file not found: {events_file}")
            return None, None, None
        
        try:
            # Load raw data
            raw = read_raw_bdf(bdf_file, preload=True, verbose=False)
            logger.info(f"Loaded raw data: {raw.info['nchan']} channels, {raw.n_times} samples")
            
            # Load events
            events_df = pd.read_csv(events_file, sep='\t')
            logger.info(f"Loaded {len(events_df)} events")
            
            return raw, events_df, bdf_file
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None, None
    
    def extract_response_times(self, events_df: pd.DataFrame, task_name: str):
        """Extract response times from events dataframe"""
        response_times = []
        trial_info = []
        
        if task_name == 'contrastChangeDetection':
            # Find target presentations and button presses
            targets = events_df[events_df['value'].isin(['right_target', 'left_target'])]
            responses = events_df[events_df['value'].isin(['right_buttonPress', 'left_buttonPress'])]
            
            for _, target in targets.iterrows():
                target_time = target['onset']
                target_type = target['value']
                
                # Find corresponding response within 5 seconds
                response_window = responses[
                    (responses['onset'] > target_time) & 
                    (responses['onset'] < target_time + 5.0)
                ]
                
                if len(response_window) > 0:
                    response = response_window.iloc[0]
                    response_time = response['onset'] - target_time
                    
                    response_times.append(response_time)
                    trial_info.append({
                        'target_type': target_type,
                        'response_type': response['value'],
                        'target_onset': target_time,
                        'response_onset': response['onset'],
                        'correct': self._is_correct_response(target_type, response['value'])
                    })
        
        elif task_name in ['seqLearning6target', 'seqLearning8target']:
            # Find learning blocks
            learning_blocks = events_df[events_df['value'].str.contains('learningBlock_', na=False)]
            
            for _, block in learning_blocks.iterrows():
                block_num = block['value'].split('_')[1]
                block_start = block['onset']
                
                # Find the response for this block
                response_row = events_df[
                    (events_df['onset'] > block_start) & 
                    (events_df['user_answer'].notna())
                ]
                
                if len(response_row) > 0:
                    response = response_row.iloc[0]
                    response_time = response['onset'] - block_start
                    
                    response_times.append(response_time)
                    trial_info.append({
                        'block_number': int(block_num),
                        'user_answer': response['user_answer'],
                        'correct_answer': response['correct_answer'],
                        'block_start': block_start,
                        'response_onset': response['onset']
                    })
        
        elif task_name == 'symbolSearch':
            # Find trial responses
            responses = events_df[events_df['value'] == 'trialResponse']
            
            for i, (_, response) in enumerate(responses.iterrows()):
                response_times.append(response['onset'])
                trial_info.append({
                    'trial_id': f"trial_{i+1}",
                    'user_answer': response['user_answer'],
                    'correct_answer': response['correct_answer'],
                    'response_onset': response['onset']
                })
        
        return response_times, trial_info
    
    def _is_correct_response(self, target_type: str, response_type: str) -> bool:
        """Check if response matches target type"""
        if target_type == 'right_target' and response_type == 'right_buttonPress':
            return True
        elif target_type == 'left_target' and response_type == 'left_buttonPress':
            return True
        return False
    
    def create_epochs(self, raw, events_df, task_name, response_times, trial_info):
        """Create epochs around stimulus/response events"""
        logger.info(f"Creating epochs for {task_name}")
        
        epochs_list = []
        epoch_info = []
        
        if task_name == 'contrastChangeDetection':
            # Create epochs around target presentations
            targets = events_df[events_df['value'].isin(['right_target', 'left_target'])]
            
            for i, (_, target) in enumerate(targets.iterrows()):
                if i < len(response_times):
                    target_time = target['onset']
                    
                    # Create epoch: 0.5s after stimulus, 2s duration
                    epoch_start = target_time + 0.5  # 500ms after stimulus
                    epoch_end = epoch_start + self.window_length
                    
                    # Extract epoch data
                    start_sample = int(epoch_start * self.sfreq)
                    end_sample = int(epoch_end * self.sfreq)
                    
                    if end_sample < raw.n_times:
                        epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                        
                        if epoch_data.shape[1] == self.window_samples:
                            epochs_list.append(epoch_data)
                            epoch_info.append({
                                'participant_id': raw.info['subject_info']['his_id'] if 'subject_info' in raw.info else 'unknown',
                                'task': task_name,
                                'trial_id': i,
                                'response_time': response_times[i],
                                'target_type': target['value'],
                                'target_onset': target_time,
                                'epoch_start': epoch_start,
                                'epoch_end': epoch_end
                            })
        
        elif task_name in ['seqLearning6target', 'seqLearning8target']:
            # Create epochs around learning blocks
            learning_blocks = events_df[events_df['value'].str.contains('learningBlock_', na=False)]
            
            for i, (_, block) in enumerate(learning_blocks.iterrows()):
                if i < len(response_times):
                    block_start = block['onset']
                    
                    # Create epoch: start of block, 2s duration
                    epoch_start = block_start
                    epoch_end = epoch_start + self.window_length
                    
                    # Extract epoch data
                    start_sample = int(epoch_start * self.sfreq)
                    end_sample = int(epoch_end * self.sfreq)
                    
                    if end_sample < raw.n_times:
                        epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                        
                        if epoch_data.shape[1] == self.window_samples:
                            epochs_list.append(epoch_data)
                            epoch_info.append({
                                'participant_id': raw.info['subject_info']['his_id'] if 'subject_info' in raw.info else 'unknown',
                                'task': task_name,
                                'trial_id': i,
                                'response_time': response_times[i],
                                'block_number': int(block['value'].split('_')[1]),
                                'block_start': block_start,
                                'epoch_start': epoch_start,
                                'epoch_end': epoch_end
                            })
        
        elif task_name == 'symbolSearch':
            # Create epochs around task start
            task_start = events_df[events_df['value'] == 'symbolSearch_start']
            
            if len(task_start) > 0:
                task_start_time = task_start.iloc[0]['onset']
                
                for i, rt in enumerate(response_times):
                    # Create epoch: response time, 2s duration
                    epoch_start = rt
                    epoch_end = epoch_start + self.window_length
                    
                    # Extract epoch data
                    start_sample = int(epoch_start * self.sfreq)
                    end_sample = int(epoch_end * self.sfreq)
                    
                    if end_sample < raw.n_times:
                        epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                        
                        if epoch_data.shape[1] == self.window_samples:
                            epochs_list.append(epoch_data)
                            epoch_info.append({
                                'participant_id': raw.info['subject_info']['his_id'] if 'subject_info' in raw.info else 'unknown',
                                'task': task_name,
                                'trial_id': i,
                                'response_time': rt,
                                'task_start': task_start_time,
                                'epoch_start': epoch_start,
                                'epoch_end': epoch_end
                            })
        
        if epochs_list:
            epochs_array = np.array(epochs_list)
            logger.info(f"Created {len(epochs_list)} epochs of shape {epochs_array.shape}")
            return epochs_array, epoch_info
        else:
            logger.warning("No epochs created")
            return None, []
    
    def extract_features(self, epochs_array):
        """Extract features from epochs"""
        logger.info(f"Extracting features from {epochs_array.shape[0]} epochs")
        
        all_features = []
        feature_names = []
        
        # 1. Statistical features
        logger.info("Extracting statistical features...")
        stat_features, stat_names = self._extract_statistical_features(epochs_array)
        all_features.append(stat_features)
        feature_names.extend(stat_names)
        
        # 2. Spectral features
        if SCIPY_AVAILABLE:
            logger.info("Extracting spectral features...")
            spectral_features, spectral_names = self._extract_spectral_features(epochs_array)
            all_features.append(spectral_features)
            feature_names.extend(spectral_names)
        
        # 3. Catch22 features
        if CATCH22_AVAILABLE:
            logger.info("Extracting catch22 features...")
            catch22_features, catch22_names = self._extract_catch22_features(epochs_array)
            all_features.append(catch22_features)
            feature_names.extend(catch22_names)
        
        # 4. Temporal features
        logger.info("Extracting temporal features...")
        temporal_features, temporal_names = self._extract_temporal_features(epochs_array)
        all_features.append(temporal_features)
        feature_names.extend(temporal_names)
        
        # Combine all features
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)
            logger.info(f"Extracted {combined_features.shape[1]} features total")
            return combined_features, feature_names
        else:
            logger.error("No features extracted!")
            return np.array([]), []
    
    def _extract_statistical_features(self, epochs_array):
        """Extract statistical features"""
        n_epochs, n_chans, n_times = epochs_array.shape
        features = []
        feature_names = []
        
        for epoch in epochs_array:
            epoch_features = []
            
            for ch in range(n_chans):
                signal = epoch[ch, :]
                
                # Basic statistics
                epoch_features.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.var(signal),
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75),
                    np.min(signal),
                    np.max(signal),
                    np.ptp(signal),
                    stats.skew(signal) if SCIPY_AVAILABLE else 0,
                    stats.kurtosis(signal) if SCIPY_AVAILABLE else 0
                ])
            
            features.append(epoch_features)
        
        # Create feature names
        stat_names = ['mean', 'std', 'var', 'median', 'q25', 'q75', 'min', 'max', 'ptp', 'skew', 'kurtosis']
        for ch in range(n_chans):
            for stat in stat_names:
                feature_names.append(f"ch{ch}_{stat}")
        
        return np.array(features), feature_names
    
    def _extract_spectral_features(self, epochs_array):
        """Extract spectral features"""
        n_epochs, n_chans, n_times = epochs_array.shape
        features = []
        feature_names = []
        
        for epoch in epochs_array:
            epoch_features = []
            
            for ch in range(n_chans):
                signal = epoch[ch, :]
                
                # Power spectral density
                freqs, psd = welch(signal, fs=self.sfreq, nperseg=min(64, n_times//2))
                
                # Band power features
                for band_name, (low, high) in self.freq_bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(psd[band_mask])
                    epoch_features.append(band_power)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                epoch_features.append(spectral_centroid)
                
                # Peak frequency
                peak_freq_idx = np.argmax(psd)
                peak_frequency = freqs[peak_freq_idx]
                epoch_features.append(peak_frequency)
            
            features.append(epoch_features)
        
        # Create feature names
        for ch in range(n_chans):
            for band_name in self.freq_bands.keys():
                feature_names.append(f"ch{ch}_{band_name}_power")
            feature_names.extend([f"ch{ch}_spectral_centroid", f"ch{ch}_peak_frequency"])
        
        return np.array(features), feature_names
    
    def _extract_catch22_features(self, epochs_array):
        """Extract catch22 features"""
        n_epochs, n_chans, n_times = epochs_array.shape
        features = []
        feature_names = []
        
        for epoch in epochs_array:
            epoch_features = []
            
            for ch in range(n_chans):
                signal = epoch[ch, :]
                
                try:
                    result = catch22_all(signal)
                    catch22_values = [result['values'][key] for key in result['names']]
                    epoch_features.extend(catch22_values)
                except:
                    # If catch22 fails, add zeros
                    epoch_features.extend([0] * 22)
            
            features.append(epoch_features)
        
        # Create feature names
        catch22_names = [
            'DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac',
            'CO_HistogramAMI_even_2_5', 'CO_trev_1_num', 'DN_OutlierInclude_p_001_mdrmd',
            'DN_OutlierInclude_n_001_mdrmd', 'SP_Summaries_welch_rect_area_5_1',
            'SP_Summaries_welch_rect_centroid', 'FC_LocalSimple_mean1_tauresrat',
            'FC_LocalSimple_mean3_stderr', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
            'CO_Embed2_Dist_tau_d_expfit_expdiff', 'SP_Summaries_welch_rect_centroid_0_1',
            'SP_Summaries_welch_rect_centroid_1_1', 'SP_Summaries_welch_rect_centroid_2_1',
            'SP_Summaries_welch_rect_centroid_3_1', 'SP_Summaries_welch_rect_centroid_4_1',
            'SP_Summaries_welch_rect_centroid_5_1', 'SP_Summaries_welch_rect_centroid_6_1',
            'SP_Summaries_welch_rect_centroid_7_1'
        ]
        
        for ch in range(n_chans):
            for name in catch22_names:
                feature_names.append(f"ch{ch}_{name}")
        
        return np.array(features), feature_names
    
    def _extract_temporal_features(self, epochs_array):
        """Extract temporal features"""
        n_epochs, n_chans, n_times = epochs_array.shape
        features = []
        feature_names = []
        
        for epoch in epochs_array:
            epoch_features = []
            
            for ch in range(n_chans):
                signal = epoch[ch, :]
                
                # Zero crossing rate
                zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
                epoch_features.append(zcr)
                
                # Mean crossing rate
                mean_val = np.mean(signal)
                mcr = np.sum(np.diff(np.sign(signal - mean_val)) != 0) / len(signal)
                epoch_features.append(mcr)
                
                # Autocorrelation
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                
                # First zero crossing
                zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                first_zero = zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)
                epoch_features.append(first_zero)
                
                # Autocorrelation decay
                decay = np.sum(autocorr[1:10])
                epoch_features.append(decay)
            
            features.append(epoch_features)
        
        # Create feature names
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_zcr", f"ch{ch}_mcr", f"ch{ch}_autocorr_zero", f"ch{ch}_autocorr_decay"])
        
        return np.array(features), feature_names

def main():
    """Main function to run real EEG feature extraction"""
    logger.info("Starting Real EEG Feature Extraction")
    
    # Initialize extractor
    data_root = "/Users/kimberly/Documents/ESE5380/eeg_challenge/local_directory"
    extractor = RealEEGFeatureExtractor(data_root)
    
    # Get participants
    participants_file = Path(data_root) / "participants.tsv"
    if participants_file.exists():
        participants_df = pd.read_csv(participants_file, sep='\t')
        participant_ids = participants_df['participant_id'].tolist()
        logger.info(f"Found {len(participant_ids)} participants")
    else:
        logger.error("Participants file not found!")
        return
    
    # Process participants
    all_features = []
    all_response_times = []
    all_epoch_info = []
    
    for participant_id in participant_ids[:3]:  # Process first 3 participants
        logger.info(f"Processing {participant_id}")
        
        # Process different tasks
        tasks = ['contrastChangeDetection', 'seqLearning8target', 'symbolSearch']
        
        for task in tasks:
            logger.info(f"Processing {participant_id} - {task}")
            
            # Load data
            raw, events_df, bdf_file = extractor.load_participant_data(participant_id, task)
            
            if raw is not None and events_df is not None:
                # Extract response times
                response_times, trial_info = extractor.extract_response_times(events_df, task)
                
                if response_times:
                    # Create epochs
                    epochs_array, epoch_info = extractor.create_epochs(raw, events_df, task, response_times, trial_info)
                    
                    if epochs_array is not None:
                        # Extract features
                        features, feature_names = extractor.extract_features(epochs_array)
                        
                        if len(features) > 0:
                            all_features.append(features)
                            all_response_times.extend(response_times[:len(features)])
                            all_epoch_info.extend(epoch_info)
                            
                            logger.info(f"Extracted {features.shape[1]} features from {features.shape[0]} epochs")
    
    # Combine all features
    if all_features:
        combined_features = np.concatenate(all_features, axis=0)
        combined_response_times = np.array(all_response_times)
        
        logger.info(f"Total features: {combined_features.shape[1]}")
        logger.info(f"Total samples: {combined_features.shape[0]}")
        
        # Save features
        np.save('real_eeg_features.npy', combined_features)
        np.save('real_response_times.npy', combined_response_times)
        
        with open('real_feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        with open('real_epoch_info.json', 'w') as f:
            json.dump(all_epoch_info, f, indent=2)
        
        # Create summary report
        report = {
            'total_features': combined_features.shape[1],
            'total_samples': combined_features.shape[0],
            'participants_processed': len(set(info['participant_id'] for info in all_epoch_info)),
            'tasks_processed': list(set(info['task'] for info in all_epoch_info)),
            'feature_categories': {
                'statistical': len([f for f in feature_names if 'mean' in f or 'std' in f]),
                'spectral': len([f for f in feature_names if 'power' in f or 'spectral' in f]),
                'catch22': len([f for f in feature_names if 'DN_' in f or 'CO_' in f or 'SP_' in f or 'FC_' in f]),
                'temporal': len([f for f in feature_names if 'zcr' in f or 'autocorr' in f])
            }
        }
        
        with open('real_extraction_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*60)
        print("REAL EEG FEATURE EXTRACTION COMPLETE!")
        print("="*60)
        print(f"Total features: {combined_features.shape[1]}")
        print(f"Total samples: {combined_features.shape[0]}")
        print(f"Participants: {report['participants_processed']}")
        print(f"Tasks: {report['tasks_processed']}")
        print("\nFiles saved:")
        print("- real_eeg_features.npy")
        print("- real_response_times.npy")
        print("- real_feature_names.json")
        print("- real_epoch_info.json")
        print("- real_extraction_report.json")
        
    else:
        logger.error("No features extracted!")

if __name__ == "__main__":
    main()
