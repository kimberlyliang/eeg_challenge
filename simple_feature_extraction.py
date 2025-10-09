#!/usr/bin/env python3
"""
This script extracts important features. We are going to try the basic statistical features first from catch22. 

Features included:
- Statistical features
- Spectral features (basic)
- Temporal features
- Channel-specific features
- Basic catch22-like features
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
import multiprocessing as mp
from functools import partial
import logging
import glob
import os
from mne.io import read_raw_bdf
from mne import create_info, Epochs, events_from_annotations
from mne.filter import filter_data
import mne

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Utility functions from data_visualization.ipynb
def get_subjects(release: int):
    """Get all subject names for a given release"""
    subject_paths = glob.glob(os.path.join(f"release{release}", "sub*"))
    subject_names = [os.path.basename(path) for path in subject_paths]
    return subject_names

def get_bdfs_for_subject(release: int, subject: str):
    """Get all BDF files for a specific subject"""
    path = f"release{release}/{subject}/eeg"
    bdf_files = glob.glob(os.path.join(path, "*.bdf"))
    return bdf_files

def get_dicts_for_release(release: int):
    """Get dictionaries mapping subjects to their BDF files"""
    r_subjects = get_subjects(release)
    
    subj_to_all_bdfs = {subj: get_bdfs_for_subject(release, subj) for subj in r_subjects}
    subj_to_split_bdfs = {}
    
    for subj in r_subjects:
        bdfs = get_bdfs_for_subject(release, subj)
        
        tasks = [os.path.basename(bdf).split("task-")[-1].replace("_eeg.bdf", "") for bdf in bdfs]
        
        task_to_bdfs = {tasks[i]: bdfs[i] for i in range(len(bdfs))}
        
        subj_to_split_bdfs[subj] = task_to_bdfs
    
    return subj_to_all_bdfs, subj_to_split_bdfs

def get_tasks_for_subj(release: int, subj: str):
    """Get all tasks for a specific subject"""
    subj_to_all_bdfs, subj_to_split_bdfs = get_dicts_for_release(release)
    return list(subj_to_split_bdfs[subj].keys())

class EEGDataLoader:
    """Load and process real EEG data from BDF files using utility functions"""
    
    def __init__(self, release: int):
        self.release = release
        self.subj_to_all_bdfs, self.subj_to_split_bdfs = get_dicts_for_release(release)
        self.subjects = get_subjects(release)
        logger.info(f"Loaded {len(self.subjects)} subjects for release {release}")
    
    def load_participant_data(self, participant_id: str, task_name: str):
        """Load EEG data for a specific participant and task"""
        logger.info(f"Loading data for {participant_id} - {task_name}")
        
        # Get BDF file for this subject and task
        if participant_id not in self.subj_to_split_bdfs:
            logger.warning(f"Subject {participant_id} not found")
            return None, None, None
        
        if task_name not in self.subj_to_split_bdfs[participant_id]:
            logger.warning(f"Task {task_name} not found for {participant_id}")
            return None, None, None
        
        bdf_file = self.subj_to_split_bdfs[participant_id][task_name]
        
        # Find corresponding events file
        events_file = bdf_file.replace("_eeg.bdf", "_events.tsv")
        
        if not os.path.exists(bdf_file):
            logger.warning(f"BDF file not found: {bdf_file}")
            return None, None, None
        
        if not os.path.exists(events_file):
            logger.warning(f"Events file not found: {events_file}")
            return None, None, None
        
        logger.info(f"Using BDF file: {bdf_file}")
        logger.info(f"Using events file: {events_file}")
        
        try:
            # Load raw data
            raw = read_raw_bdf(bdf_file, preload=True, verbose=False)
            logger.info(f"Loaded raw data: {raw.info['nchan']} channels, {raw.n_times} samples")
            
            # Check Cz channel specifically
            if 'Cz' in raw.ch_names:
                cz_idx = raw.ch_names.index('Cz')
                cz_data = raw.get_data()[cz_idx, :]
                logger.info(f"Cz channel (index {cz_idx}): mean={np.mean(cz_data):.3f}, std={np.std(cz_data):.3f}, range=[{np.min(cz_data):.3f}, {np.max(cz_data):.3f}]")
                logger.info(f"Cz has NaN: {np.isnan(cz_data).any()}, has Inf: {np.isinf(cz_data).any()}")
            else:
                logger.warning("Cz channel not found in data")
                logger.info(f"Available channels: {raw.ch_names[:10]}...")  # Show first 10 channels
            
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
        
        if 'contrastChangeDetection' in task_name:
            # Look for target events (left_target=8, right_target=9) and button responses (left_buttonPress=12, right_buttonPress=13)
            target_events = events_df[events_df['event_code'].isin([8, 9])]  # left_target=8, right_target=9
            button_events = events_df[events_df['event_code'].isin([12, 13])]  # left_buttonPress=12, right_buttonPress=13
            
            logger.info(f"Found {len(target_events)} target events and {len(button_events)} button events")
            
            for _, target in target_events.iterrows():
                # Find corresponding button press within reasonable time window
                button_candidates = button_events[
                    (button_events['onset'] > target['onset']) & 
                    (button_events['onset'] <= target['onset'] + 5.0)  # 5 second window
                ]
                
                if len(button_candidates) > 0:
                    button = button_candidates.iloc[0]
                    rt = button['onset'] - target['onset']
                    response_times.append(rt)
                    trial_info.append({
                        'target_onset': target['onset'],
                        'button_onset': button['onset'],
                        'target_value': target['value'],
                        'button_value': button['value'],
                        'target_code': target['event_code'],
                        'button_code': button['event_code'],
                        'response_time': rt
                    })
                    logger.debug(f"Response time: {rt:.3f}s for target at {target['onset']:.3f}s")
                else:
                    logger.debug(f"No button press found for target at {target['onset']:.3f}s")
        
        elif task_name == 'symbolSearch':
            # Look for trialResponse events
            response_events = events_df[events_df['value'] == 'trialResponse']
            
            for _, response in response_events.iterrows():
                # For symbolSearch, we'll use a fixed response time based on the task design
                # The task has a fixed timing, so we'll use a reasonable estimate
                response_times.append(2.0)  # Estimated response time for symbol search
                trial_info.append({
                    'response_onset': response['onset'],
                    'user_answer': response.get('user_answer', 'n/a'),
                    'correct_answer': response.get('correct_answer', 'n/a')
                })
        
        elif task_name in ['seqLearning6target', 'seqLearning8target']:
            # Look for dot ON events as stimulus markers
            dot_on_events = events_df[events_df['value'].str.contains('dot_.*_ON', na=False)]
            
            for _, dot_event in dot_on_events.iterrows():
                # For sequence learning, we'll use a fixed response time
                # This is a passive learning task, so response time is less relevant
                response_times.append(1.0)  # Estimated response time for sequence learning
                trial_info.append({
                    'dot_onset': dot_event['onset'],
                    'dot_value': dot_event['value']
                })
        
        elif task_name == 'surroundSupp':
            # Look for stimulus events
            stimulus_events = events_df[events_df['value'].str.contains('stimulus', na=False)]
            
            for _, stimulus in stimulus_events.iterrows():
                response_times.append(0.5)  # Estimated response time for surround suppression
                trial_info.append({
                    'stimulus_onset': stimulus['onset'],
                    'stimulus_value': stimulus['value']
                })
        
        else:
            # For other tasks, use a default approach
            logger.warning(f"Unknown task: {task_name}, using default response time extraction")
            # Use all events as markers with default response time
            for _, event in events_df.iterrows():
                if event['value'] not in ['break cnt', 'n/a']:
                    response_times.append(1.0)  # Default response time
                    trial_info.append({
                        'event_onset': event['onset'],
                        'event_value': event['value']
                    })
        
        logger.info(f"Extracted {len(response_times)} response times for {task_name}")
        if response_times:
            logger.info(f"Response time range: {min(response_times):.3f}s to {max(response_times):.3f}s")
            logger.info(f"Mean response time: {np.mean(response_times):.3f}s")
        return response_times, trial_info
    
    def create_epochs(self, raw, events_df: pd.DataFrame, task_name: str, response_times: List[float], 
                     trial_info: List[dict], epoch_length: float = 2.0):
        """Create epochs around response events"""
        epochs_array = []
        epoch_info = []
        
        # Set up epoch parameters
        sfreq = raw.info['sfreq']
        epoch_samples = int(epoch_length * sfreq)
        
        if 'contrastChangeDetection' in task_name:
            # Create epochs around target events (left_target=8, right_target=9)
            target_events = events_df[events_df['event_code'].isin([8, 9])]
            
            for i, (_, target) in enumerate(target_events.iterrows()):
                if i >= len(response_times):
                    break
                    
                start_sample = int(target['onset'] * sfreq)
                end_sample = start_sample + epoch_samples
                
                if end_sample < raw.n_times:
                    epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                    epochs_array.append(epoch_data)
                    epoch_info.append({
                        'participant': 'unknown',
                        'task': task_name,
                        'trial': i,
                        'response_time': response_times[i] if i < len(response_times) else 0.0,
                        'target_code': target['event_code'],
                        'target_value': target['value']
                    })
        
        elif task_name == 'symbolSearch':
            # Create epochs around response events
            response_events = events_df[events_df['value'] == 'trialResponse']
            
            for i, (_, response) in enumerate(response_events.iterrows()):
                if i >= 50:  # Limit to first 50 trials to avoid memory issues
                    break
                    
                start_sample = int(response['onset'] * sfreq) - epoch_samples // 2
                end_sample = start_sample + epoch_samples
                
                if start_sample >= 0 and end_sample < raw.n_times:
                    epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                    epochs_array.append(epoch_data)
                    epoch_info.append({
                        'participant': 'unknown',
                        'task': task_name,
                        'trial': i,
                        'response_time': response_times[i] if i < len(response_times) else 2.0
                    })
        
        elif task_name in ['seqLearning6target', 'seqLearning8target']:
            # Create epochs around dot ON events
            dot_on_events = events_df[events_df['value'].str.contains('dot_.*_ON', na=False)]
            
            for i, (_, dot_event) in enumerate(dot_on_events.iterrows()):
                if i >= 100:  # Limit to first 100 trials to avoid memory issues
                    break
                    
                start_sample = int(dot_event['onset'] * sfreq)
                end_sample = start_sample + epoch_samples
                
                if end_sample < raw.n_times:
                    epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                    epochs_array.append(epoch_data)
                    epoch_info.append({
                        'participant': 'unknown',
                        'task': task_name,
                        'trial': i,
                        'response_time': response_times[i] if i < len(response_times) else 1.0
                    })
        
        else:
            # For other tasks, create epochs around all relevant events
            relevant_events = events_df[~events_df['value'].isin(['break cnt', 'n/a'])]
            
            for i, (_, event) in enumerate(relevant_events.iterrows()):
                if i >= 50:  # Limit to first 50 trials to avoid memory issues
                    break
                    
                start_sample = int(event['onset'] * sfreq)
                end_sample = start_sample + epoch_samples
                
                if end_sample < raw.n_times:
                    epoch_data = raw.get_data(start=start_sample, stop=end_sample)
                    epochs_array.append(epoch_data)
                    epoch_info.append({
                        'participant': 'unknown',
                        'task': task_name,
                        'trial': i,
                        'response_time': response_times[i] if i < len(response_times) else 1.0
                    })
        
        if epochs_array:
            epochs_array = np.array(epochs_array)
            logger.info(f"Created {epochs_array.shape[0]} epochs of shape {epochs_array.shape[1:]}")
            return epochs_array, epoch_info
        else:
            logger.warning("No epochs created")
            return None, None

class SimpleEEGFeatureExtractor:
    """Simplified EEG feature extraction class"""
    
    def __init__(self, sfreq=100, n_chans=129, n_times=200):
        self.sfreq = sfreq
        self.n_chans = n_chans
        self.n_times = n_times
        self.feature_names = []
        
        # Frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13), # this one is the one we care about I think
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        logger.info(f"Initialized SimpleEEGFeatureExtractor with {n_chans} channels, {n_times} time points")
    
    def extract_all_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract all features from EEG data
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_chans, n_times)
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        logger.info(f"Extracting features from {eeg_data.shape[0]} samples")
        
        all_features = []
        feature_names = []
        
        # # 1 statistical features
        # logger.info("Extracting statistical features...")
        # stat_features, stat_names = self._extract_statistical_features(eeg_data)
        # all_features.append(stat_features)
        # feature_names.extend(stat_names)
        
        # # 2. basic spectral features
        # logger.info("Extracting spectral features...")
        # spectral_features, spectral_names = self._extract_spectral_features(eeg_data)
        # all_features.append(spectral_features)
        # feature_names.extend(spectral_names)
        
        # # 3. temporal features
        # logger.info("Extracting temporal features...")
        # temporal_features, temporal_names = self._extract_temporal_features(eeg_data)
        # all_features.append(temporal_features)
        # feature_names.extend(temporal_names)
        
        # # 4. channel-specific features
        # logger.info("Extracting channel-specific features...")
        # channel_features, channel_names = self._extract_channel_features(eeg_data)
        # all_features.append(channel_features)
        # feature_names.extend(channel_names)
        
        # 5. basic catch22-like features
        logger.info("Extracting catch22-like features...")
        catch22_features, catch22_names = self._extract_catch22_like_features(eeg_data)
        all_features.append(catch22_features)
        feature_names.extend(catch22_names)
        
        # 6. Entropy features 
        # logger.info("Extracting entropy features...")
        # entropy_features, entropy_names = self._extract_entropy_features(eeg_data)
        # all_features.append(entropy_features)
        # feature_names.extend(entropy_names)
        
        # # 7. Connectivity features (basic)
        # logger.info("Extracting connectivity features...")
        # conn_features, conn_names = self._extract_connectivity_features(eeg_data)
        # all_features.append(conn_features)
        # feature_names.extend(conn_names)
        
        # Combine all features
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)
            self.feature_names = feature_names
            logger.info(f"Extracted {combined_features.shape[1]} features total")
            return combined_features
        else:
            logger.error("No features extracted!")
            return np.array([])
    
    def _extract_statistical_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # Basic statistics
                sample_features.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.var(signal),
                    # np.median(signal),
                    # np.percentile(signal, 25), # not needed
                    # np.percentile(signal, 75), # not needed
                    np.min(signal),
                    np.max(signal),
                    np.ptp(signal),  # peak-to-peak
                    # self._skewness(signal), # not needed 
                    # self._kurtosis(signal)
                ])
                
                # Higher-order moments
                sample_features.extend([
                    np.mean(np.abs(signal)),
                    np.mean(np.square(signal)),
                    np.sqrt(np.mean(np.square(signal))),  # RMS
                    np.mean(np.abs(np.diff(signal))),  # mean absolute difference
                    np.std(np.diff(signal))  # std of differences
                ])
            
            features.append(sample_features)
        
        # feature names
        stat_names = ['mean', 'std', 'var', 'median', 'q25', 'q75', 'min', 'max', 'ptp', 'skew', 'kurtosis',
                     'mean_abs', 'mean_square', 'rms', 'mean_abs_diff', 'std_diff']
        for ch in range(n_chans):
            for stat in stat_names:
                feature_names.append(f"ch{ch}_{stat}")
        
        return np.array(features), feature_names
    
    def _extract_spectral_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract basic spectral features using FFT"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        # FFT parameters
        freqs = np.fft.fftfreq(n_times, 1/self.sfreq) # with nyquist parameters here I think
        freqs = freqs[:n_times//2]  # only positive frequencies
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # FFT
                fft_signal = np.fft.fft(signal)
                power_spectrum = np.abs(fft_signal[:n_times//2]) ** 2
                
                # Band power features
                for band_name, (low, high) in self.freq_bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(power_spectrum[band_mask])
                    sample_features.append(band_power)
                
                # spectral centroid
                spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                sample_features.append(spectral_centroid)
                
                # spectral rolloff (85% of power)
                cumsum_power = np.cumsum(power_spectrum)
                rolloff_idx = np.where(cumsum_power >= 0.85 * cumsum_power[-1])[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                sample_features.append(spectral_rolloff)
                
                # spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
                sample_features.append(spectral_bandwidth)
                
                # peak frequency
                peak_freq_idx = np.argmax(power_spectrum)
                peak_frequency = freqs[peak_freq_idx]
                sample_features.append(peak_frequency)
            
            features.append(sample_features)
        
        for ch in range(n_chans):
            for band_name in self.freq_bands.keys():
                feature_names.append(f"ch{ch}_{band_name}_power")
            feature_names.extend([f"ch{ch}_spectral_centroid", f"ch{ch}_spectral_rolloff", 
                                f"ch{ch}_spectral_bandwidth", f"ch{ch}_peak_frequency"])
        
        return np.array(features), feature_names
    
    def _extract_temporal_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract temporal features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # autocorrelation features
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # zero crossing of autocorrelation
                zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                if len(zero_crossings) > 0:
                    first_zero = zero_crossings[0]
                    sample_features.append(first_zero)
                else:
                    sample_features.append(len(autocorr))
                
                # autocorrelation decay
                decay = np.sum(autocorr[1:10])
                sample_features.append(decay)
                
                # trend features
                x = np.arange(len(signal))
                slope, intercept, r_value, p_value, std_err = self._linear_regression(x, signal)
                sample_features.extend([slope, r_value, p_value])
                
                # zero crossing rate
                zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
                sample_features.append(zcr)
                
                # mean crossing rate
                mean_val = np.mean(signal)
                mcr = np.sum(np.diff(np.sign(signal - mean_val)) != 0) / len(signal)
                sample_features.append(mcr)
            
            features.append(sample_features)
        
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_autocorr_zero", f"ch{ch}_autocorr_decay", 
                                f"ch{ch}_trend_slope", f"ch{ch}_trend_r", f"ch{ch}_trend_p",
                                f"ch{ch}_zcr", f"ch{ch}_mcr"])
        
        return np.array(features), feature_names
    
    def _extract_channel_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract channel-specific features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Global features across all channels
            all_channels = eeg_data[i, :, :].flatten()
            sample_features.extend([
                np.mean(all_channels),
                np.std(all_channels),
                np.var(all_channels),
                np.median(all_channels),
                np.ptp(all_channels)
            ])
            
            # Channel-specific features
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # Signal energy
                energy = np.sum(signal ** 2)
                sample_features.append(energy)
                
                # Signal power
                power = np.mean(signal ** 2)
                sample_features.append(power)
                
                # Signal-to-noise ratio (simplified)
                signal_power = np.mean(signal ** 2)
                noise_power = np.var(signal - np.mean(signal))
                snr = signal_power / (noise_power + 1e-10)
                sample_features.append(snr)
            
            features.append(sample_features)
        
        feature_names.extend(['global_mean', 'global_std', 'global_var', 'global_median', 'global_ptp'])
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_energy", f"ch{ch}_power", f"ch{ch}_snr"])
        
        return np.array(features), feature_names
    
    def _extract_catch22_like_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract catch22-like features (simplified versions)"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # basic catch22-like features
                sample_features.extend([
                    self._mean_abs_change(signal),
                    self._mean_change(signal),
                    self._mean_second_derivative_central(signal),
                    self._variance(signal),
                    self._standard_deviation(signal),
                    self._skewness(signal),
                    self._kurtosis(signal),
                    self._root_mean_square(signal),
                    self._linear_trend_slope(signal),
                    self._autocorr_lag1(signal),
                    self._autocorr_lag2(signal),
                    self._autocorr_lag3(signal),
                    self._autocorr_lag4(signal),
                    self._autocorr_lag5(signal),
                    self._first_min_ac(signal),
                    self._first_zero_ac(signal),
                    self._mean_abs_derivative(signal),
                    self._mean_derivative(signal),
                    self._median(signal),
                    self._min(signal),
                    self._max(signal),
                    self._range(signal)
                ])
            
            features.append(sample_features)
        
        # Create feature names
        catch22_names = [
            'mean_abs_change', 'mean_change', 'mean_second_derivative_central', 'variance', 'standard_deviation',
            'skewness', 'kurtosis', 'root_mean_square', 'linear_trend_slope', 'autocorr_lag1', 'autocorr_lag2',
            'autocorr_lag3', 'autocorr_lag4', 'autocorr_lag5', 'first_min_ac', 'first_zero_ac',
            'mean_abs_derivative', 'mean_derivative', 'median', 'min', 'max', 'range'
        ]
        
        for ch in range(n_chans):
            for name in catch22_names:
                feature_names.append(f"ch{ch}_{name}")
        
        return np.array(features), feature_names
    
    def _extract_entropy_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract entropy features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                # shannon entropy
                hist, _ = np.histogram(signal, bins=50)
                hist = hist / np.sum(hist)
                shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))
                sample_features.append(shannon_entropy)
                
                # approximate entropy (simplified)
                approx_entropy = self._approximate_entropy(signal, m=2, r=0.2)
                sample_features.append(approx_entropy)
                
                # sample entropy (simplified)
                sample_entropy = self._sample_entropy(signal, m=2, r=0.2)
                sample_features.append(sample_entropy)
            
            features.append(sample_features)
        
        # Create feature names
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_shannon_entropy", f"ch{ch}_approx_entropy", f"ch{ch}_sample_entropy"])
        
        return np.array(features), feature_names
    
    def _extract_connectivity_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract basic connectivity features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        # Select subset of channels for connectivity (to avoid memory issues)
        n_conn_chans = min(10, n_chans)
        conn_indices = np.linspace(0, n_chans-1, n_conn_chans, dtype=int)
        
        for i in range(n_samples):
            sample_features = []
            
            # Calculate connectivity between selected channels
            for ch1 in range(n_conn_chans):
                for ch2 in range(ch1 + 1, n_conn_chans):
                    idx1, idx2 = conn_indices[ch1], conn_indices[ch2]
                    
                    # Cross-correlation
                    corr = np.corrcoef(eeg_data[i, idx1, :], eeg_data[i, idx2, :])[0, 1]
                    sample_features.append(corr)
                    
                    # Coherence (simplified)
                    fft1 = np.fft.fft(eeg_data[i, idx1, :])
                    fft2 = np.fft.fft(eeg_data[i, idx2, :])
                    coherence = np.abs(np.mean(fft1 * np.conj(fft2))) / (np.sqrt(np.mean(np.abs(fft1)**2) * np.mean(np.abs(fft2)**2)) + 1e-10)
                    sample_features.append(coherence)
            
            features.append(sample_features)
        
        # Create feature names
        for ch1 in range(n_conn_chans):
            for ch2 in range(ch1 + 1, n_conn_chans):
                idx1, idx2 = conn_indices[ch1], conn_indices[ch2]
                feature_names.extend([f"corr_ch{idx1}_ch{idx2}", f"coh_ch{idx1}_ch{idx2}"])
        
        return np.array(features), feature_names
    
    # Helper methods for catch22-like features
    def _mean_abs_change(self, x):
        return np.mean(np.abs(np.diff(x)))
    
    def _mean_change(self, x):
        return np.mean(np.diff(x))
    
    def _mean_second_derivative_central(self, x):
        return np.mean(np.diff(np.diff(x)))
    
    def _variance(self, x):
        return np.var(x)
    
    def _standard_deviation(self, x):
        return np.std(x)
    
    def _skewness(self, x):
        mean_x = np.mean(x)
        std_x = np.std(x)
        return np.mean(((x - mean_x) / std_x) ** 3)
    
    def _kurtosis(self, x):
        mean_x = np.mean(x)
        std_x = np.std(x)
        return np.mean(((x - mean_x) / std_x) ** 4) - 3
    
    def _root_mean_square(self, x):
        return np.sqrt(np.mean(x ** 2))
    
    def _linear_trend_slope(self, x):
        n = len(x)
        y = np.arange(n)
        return np.corrcoef(x, y)[0, 1]
    
    def _autocorr_lag1(self, x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    
    def _autocorr_lag2(self, x):
        return np.corrcoef(x[:-2], x[2:])[0, 1]
    
    def _autocorr_lag3(self, x):
        return np.corrcoef(x[:-3], x[3:])[0, 1]
    
    def _autocorr_lag4(self, x):
        return np.corrcoef(x[:-4], x[4:])[0, 1]
    
    def _autocorr_lag5(self, x):
        return np.corrcoef(x[:-5], x[5:])[0, 1]
    
    def _first_min_ac(self, x):
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        return np.argmin(autocorr[1:]) + 1
    
    def _first_zero_ac(self, x):
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        return zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)
    
    def _mean_abs_derivative(self, x):
        return np.mean(np.abs(np.diff(x)))
    
    def _mean_derivative(self, x):
        return np.mean(np.diff(x))
    
    def _median(self, x):
        return np.median(x)
    
    def _min(self, x):
        return np.min(x)
    
    def _max(self, x):
        return np.max(x)
    
    def _range(self, x):
        return np.ptp(x)
    
    def _linear_regression(self, x, y):
        """Simple linear regression"""
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate correlation coefficient
        r = np.corrcoef(x, y)[0, 1]
        
        # Calculate p-value (simplified)
        p_value = 0.05  # Placeholder
        
        # Calculate standard error
        std_err = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
        
        return slope, intercept, r, p_value, std_err
    
    def _approximate_entropy(self, data, m, r):
        """Calculate approximate entropy (simplified)"""
        N = len(data)
        if N < m + 1:
            return 0
        
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _approximate_entropy_single(data, m, r):
            N = len(data)
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                template_i = data[i:i + m]
                for j in range(N - m + 1):
                    template_j = data[j:j + m]
                    if _maxdist(template_i, template_j, m) <= r:
                        C[i] += 1.0
            C = C / (N - m + 1.0)
            phi = np.mean(np.log(C + 1e-10))
            return phi
        
        return _approximate_entropy_single(data, m, r) - _approximate_entropy_single(data, m + 1, r)
    
    def _sample_entropy(self, data, m, r):
        """Calculate sample entropy (simplified)"""
        N = len(data)
        if N < m + 1:
            return 0
        
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _sample_entropy_single(data, m, r):
            N = len(data)
            B = 0.0
            A = 0.0
            
            for i in range(N - m):
                template_i = data[i:i + m]
                for j in range(i + 1, N - m):
                    template_j = data[j:j + m]
                    if _maxdist(template_i, template_j, m) <= r:
                        B += 1.0
                        if _maxdist(data[i:i + m + 1], data[j:j + m + 1], m + 1) <= r:
                            A += 1.0
            
            if B == 0:
                return 0
            return -np.log(A / B + 1e-10)
        
        return _sample_entropy_single(data, m, r)

def find_data_directory(release: int):
    """Find the EEG data directory for a given release"""
    release_dir = f"release{release}"
    
    if os.path.exists(release_dir):
        logger.info(f"Found data directory: {release_dir}")
        return release_dir
    
    return None

def run_feature_extraction(release: int = 1):
    """Run the feature extraction pipeline with real EEG data"""
    logger.info("Starting feature extraction pipeline with real EEG data")
    start_time = datetime.now()
    
    # Initialize data loader with release number
    data_loader = EEGDataLoader(release)
    
    # Get participant IDs
    participant_ids = data_loader.subjects
    logger.info(f"Found {len(participant_ids)} participants")
    
    # Initialize feature extractor
    extractor = SimpleEEGFeatureExtractor(sfreq=100, n_chans=129, n_times=200)
    
    # Collect all features and response times
    all_features = []
    all_response_times = []
    all_epoch_info = []
    
    # Process first few participants to avoid memory issues
    max_participants = 3
    # Get available tasks from the first subject
    if participant_ids:
        available_tasks = get_tasks_for_subj(release, participant_ids[0])
        logger.info(f"Available tasks: {available_tasks}")
        
        # Prioritize contrast change detection since that's where we care about response times
        contrast_tasks = [task for task in available_tasks if 'contrastChangeDetection' in task]
        if contrast_tasks:
            # Use the first contrast change detection run
            tasks_to_process = [contrast_tasks[0]] + [task for task in available_tasks if 'contrastChangeDetection' not in task][:2]
            logger.info(f"Prioritizing {contrast_tasks[0]} for real response times")
        else:
            tasks_to_process = available_tasks[:3]  # Process first 3 available tasks
            logger.warning("contrastChangeDetection not available - using other tasks with estimated response times")
    else:
        tasks_to_process = ['symbolSearch']  # Fallback
    
    for participant_id in participant_ids[:max_participants]:
        logger.info(f"Processing {participant_id}")
        
        for task in tasks_to_process:
            logger.info(f"Processing {participant_id} - {task}")
            
            # Load data
            raw, events_df, bdf_file = data_loader.load_participant_data(participant_id, task)
            
            if raw is not None and events_df is not None:
                # Extract response times
                response_times, trial_info = data_loader.extract_response_times(events_df, task)
                
                if response_times:
                    logger.info(f"Task {task}: Got {len(response_times)} response times")
                    if 'contrastChangeDetection' in task:
                        logger.info(f"Real response times from contrast change detection")
                    else:
                        logger.info(f"Estimated response times from {task}")
                    
                    # Create epochs
                    epochs_array, epoch_info = data_loader.create_epochs(raw, events_df, task, response_times, trial_info)
                    
                    if epochs_array is not None:
                        # Check Cz channel in epochs
                        if epochs_array.shape[1] >= 129:  # Assuming 129 channels
                            cz_epochs = epochs_array[:, 128, :]  # Channel 128 (Cz)
                            logger.info(f"Cz epochs: shape={cz_epochs.shape}, mean={np.mean(cz_epochs):.3f}, std={np.std(cz_epochs):.3f}")
                            logger.info(f"Cz epochs has NaN: {np.isnan(cz_epochs).any()}, has Inf: {np.isinf(cz_epochs).any()}")
                        
                        try:
                            # Extract features
                            features = extractor.extract_all_features(epochs_array)
                            
                            if features is not None and len(features) > 0:
                                # Check if features is a numpy array
                                if isinstance(features, np.ndarray):
                                    all_features.append(features)
                                    all_response_times.extend(response_times[:len(features)])
                                    all_epoch_info.extend(epoch_info)
                                    
                                    logger.info(f"Extracted {features.shape[1]} features from {features.shape[0]} epochs")
                                else:
                                    logger.warning(f"Features is not a numpy array: {type(features)}")
                            else:
                                logger.warning(f"No features extracted for {participant_id} - {task}")
                        except Exception as e:
                            logger.error(f"Error extracting features for {participant_id} - {task}: {e}")
                            continue
    
    # Combine all features
    if all_features:
        combined_features = np.vstack(all_features)
        combined_response_times = np.array(all_response_times)
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        logger.info(f"Combined response times shape: {combined_response_times.shape}")
        
        # Save features
        logger.info("Saving features...")
        np.save('eeg_features.npy', combined_features)
        np.save('response_times.npy', combined_response_times)
        
        with open('feature_names.json', 'w') as f:
            json.dump(extractor.feature_names, f, indent=2)
        
        # Basic analysis
        logger.info("Performing basic analysis...")
        
        # Check for NaN or infinite values
        nan_count = np.isnan(combined_features).sum()
        inf_count = np.isinf(combined_features).sum()
        
        logger.info(f"NaN values: {nan_count}")
        logger.info(f"Infinite values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            logger.warning("Cleaning data...")
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature statistics
        feature_stats = {
            'mean': np.mean(combined_features, axis=0),
            'std': np.std(combined_features, axis=0),
            'min': np.min(combined_features, axis=0),
            'max': np.max(combined_features, axis=0)
        }
        
        # Save statistics
        with open('feature_stats.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in feature_stats.items()}, f, indent=2)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # Feature importance (correlation with target)
        # Make sure we have the same number of samples for features and response times
        min_samples = min(combined_features.shape[0], len(combined_response_times))
        if combined_features.shape[0] != len(combined_response_times):
            logger.warning(f"Mismatch: {combined_features.shape[0]} features vs {len(combined_response_times)} response times. Using {min_samples} samples.")
            combined_features = combined_features[:min_samples]
            combined_response_times = combined_response_times[:min_samples]
        
        # Debug: Check response time distribution
        logger.info(f"Response time statistics:")
        logger.info(f"  Mean: {np.mean(combined_response_times):.3f}s")
        logger.info(f"  Std: {np.std(combined_response_times):.3f}s")
        logger.info(f"  Min: {np.min(combined_response_times):.3f}s")
        logger.info(f"  Max: {np.max(combined_response_times):.3f}s")
        logger.info(f"  Unique values: {len(np.unique(combined_response_times))}")
        
        # Check for issues before correlation calculation
        logger.info(f"Before correlation calculation:")
        logger.info(f"  Features shape: {combined_features.shape}")
        logger.info(f"  Response times shape: {combined_response_times.shape}")
        logger.info(f"  Features has NaN: {np.isnan(combined_features).any()}")
        logger.info(f"  Response times has NaN: {np.isnan(combined_response_times).any()}")
        logger.info(f"  Response times std: {np.std(combined_response_times):.6f}")
        
        # Check if response times have variance
        if np.std(combined_response_times) == 0:
            logger.warning("Response times have no variance - cannot calculate meaningful correlations")
            # Create synthetic correlations for demonstration
            logger.info("Creating synthetic correlations for demonstration purposes")
            # Use the first feature as a synthetic response time predictor
            if combined_features.shape[1] > 0:
                synthetic_rt = combined_features[:, 0]  # Use first feature as synthetic response time
                correlations = np.abs(np.corrcoef(combined_features.T, synthetic_rt)[-1, :-1])
            else:
                correlations = np.zeros(combined_features.shape[1])
        else:
            try:
                correlations = np.corrcoef(combined_features.T, combined_response_times)[-1, :-1]
                correlations = np.abs(correlations)
                
                # Handle NaN correlations
                if np.isnan(correlations).any():
                    logger.warning(f"Found {np.isnan(correlations).sum()} NaN correlations")
                    correlations = np.nan_to_num(correlations, nan=0.0)
            except Exception as e:
                logger.error(f"Error calculating correlations: {e}")
                correlations = np.zeros(combined_features.shape[1])
        
        # Debug: Check correlation statistics
        logger.info(f"Correlation statistics:")
        logger.info(f"  Mean correlation: {np.mean(correlations):.3f}")
        logger.info(f"  Max correlation: {np.max(correlations):.3f}")
        logger.info(f"  Features with correlation > 0.1: {np.sum(correlations > 0.1)}")
        logger.info(f"  Features with correlation > 0.5: {np.sum(correlations > 0.5)}")
        logger.info(f"  NaN correlations: {np.isnan(correlations).sum()}")
        
        # Check Cz-specific features if available
        if hasattr(extractor, 'feature_names') and extractor.feature_names:
            cz_features = [i for i, name in enumerate(extractor.feature_names) if 'ch128_' in name]  # Assuming Cz is channel 128
            if cz_features:
                cz_correlations = correlations[cz_features]
                logger.info(f"Cz features (ch128): {len(cz_features)} features")
                logger.info(f"  Cz mean correlation: {np.mean(cz_correlations):.3f}")
                logger.info(f"  Cz max correlation: {np.max(cz_correlations):.3f}")
                logger.info(f"  Cz features with correlation > 0.1: {np.sum(cz_correlations > 0.1)}")
            else:
                logger.info("No Cz features found (ch128)")
        
        # Top 20 most correlated features
        if len(correlations) > 0:
            top_indices = np.argsort(correlations)[-20:]
            top_correlations = correlations[top_indices]
            if hasattr(extractor, 'feature_names') and extractor.feature_names:
                top_names = [extractor.feature_names[i] for i in top_indices]
            else:
                top_names = [f"feature_{i}" for i in top_indices]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_correlations)), top_correlations)
            plt.yticks(range(len(top_correlations)), top_names)
            plt.xlabel('Absolute Correlation with Response Time')
            plt.title('Top 20 Most Correlated Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            logger.warning("No correlations to plot")
        
        # Feature distributions
        plt.figure(figsize=(15, 10))
        for i in range(min(20, combined_features.shape[1])):
            plt.subplot(4, 5, i + 1)
            plt.hist(combined_features[:, i], bins=50, alpha=0.7)
            if hasattr(extractor, 'feature_names') and extractor.feature_names and i < len(extractor.feature_names):
                plt.title(f'{extractor.feature_names[i][:20]}...')
            else:
                plt.title(f'feature_{i}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary report
        end_time = datetime.now()
        duration = end_time - start_time
        
        report = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'n_samples': len(combined_response_times),
            'n_features': combined_features.shape[1],
            'n_participants_processed': max_participants,
            'tasks_processed': tasks_to_process,
            'feature_categories': {
                'statistical': False,
                'spectral': False,
                'temporal': False,
                'channel_specific': False,
                'catch22_like': True,
                'entropy': False,
                'connectivity': False
            },
            'data_quality': {
                'nan_values': int(nan_count),
                'infinite_values': int(inf_count)
            }
        }
        
        with open('feature_extraction_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Feature extraction complete! Duration: {duration}")
        logger.info(f"Total features: {combined_features.shape[1]}")
        logger.info(f"Number of samples: {len(combined_response_times)}")
        
        feature_names = extractor.feature_names if hasattr(extractor, 'feature_names') and extractor.feature_names else []
        return combined_features, combined_response_times, feature_names
    else:
        logger.error("No features extracted!")
        return np.array([]), np.array([]), []

def main(release: int = 1):
    """Main function to run the feature extraction pipeline"""
    logger.info("Starting Simple EEG Feature Extraction Pipeline with Real Data")
    
    try:
        # Run the feature extraction
        features, response_times, feature_names = run_feature_extraction(release)
        
        if len(features) > 0:
            print("\n" + "="*60)
            print("FEATURE EXTRACTION COMPLETE!")
            print("="*60)
            print(f"Total features extracted: {features.shape[1]}")
            print(f"Number of samples: {len(response_times)}")
            print("\nFiles saved:")
            print("- eeg_features.npy: All extracted features")
            print("- response_times.npy: Target response times")
            print("- feature_names.json: Names of all features")
            print("- feature_stats.json: Feature statistics")
            print("- feature_extraction_report.json: Summary report")
            print("- feature_importance.png: Feature importance plot")
            print("- feature_distributions.png: Feature distribution plots")
            
            # Show some example features
            print(f"\nExample features:")
            for i in range(min(10, len(feature_names))):
                print(f"  {i+1}. {feature_names[i]}")
        else:
            print("\n" + "="*60)
            print("NO FEATURES EXTRACTED!")
            print("="*60)
            print("Check the logs for error messages.")
            print("Make sure the data path is correct and files exist.")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        print(f"\nError: {e}")
        print("Check the logs for more details.")
        raise

if __name__ == "__main__":
    # You can now run the script in several ways:
    
    # 1. Use default release 1 (recommended)
    main()
    
    # 2. Specify release number
    # main(release=2)
    
    # 3. Use in code
    # from simple_feature_extraction import run_feature_extraction
    # features, response_times, names = run_feature_extraction(release=1)
