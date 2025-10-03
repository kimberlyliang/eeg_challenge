#!/usr/bin/env python3
"""
Simplified Feature Extraction Pipeline for EEG Challenge 1

This script extracts important features without requiring heavy dependencies.
It can run overnight and provides comprehensive feature analysis.

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

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            'alpha': (8, 13),
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
        
        # 1. Statistical features
        logger.info("Extracting statistical features...")
        stat_features, stat_names = self._extract_statistical_features(eeg_data)
        all_features.append(stat_features)
        feature_names.extend(stat_names)
        
        # 2. Spectral features (basic)
        logger.info("Extracting spectral features...")
        spectral_features, spectral_names = self._extract_spectral_features(eeg_data)
        all_features.append(spectral_features)
        feature_names.extend(spectral_names)
        
        # 3. Temporal features
        logger.info("Extracting temporal features...")
        temporal_features, temporal_names = self._extract_temporal_features(eeg_data)
        all_features.append(temporal_features)
        feature_names.extend(temporal_names)
        
        # 4. Channel-specific features
        logger.info("Extracting channel-specific features...")
        channel_features, channel_names = self._extract_channel_features(eeg_data)
        all_features.append(channel_features)
        feature_names.extend(channel_names)
        
        # 5. Basic catch22-like features
        logger.info("Extracting catch22-like features...")
        catch22_features, catch22_names = self._extract_catch22_like_features(eeg_data)
        all_features.append(catch22_features)
        feature_names.extend(catch22_names)
        
        # 6. Entropy features
        logger.info("Extracting entropy features...")
        entropy_features, entropy_names = self._extract_entropy_features(eeg_data)
        all_features.append(entropy_features)
        feature_names.extend(entropy_names)
        
        # 7. Connectivity features (basic)
        logger.info("Extracting connectivity features...")
        conn_features, conn_names = self._extract_connectivity_features(eeg_data)
        all_features.append(conn_features)
        feature_names.extend(conn_names)
        
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
                    np.median(signal),
                    np.percentile(signal, 25),
                    np.percentile(signal, 75),
                    np.min(signal),
                    np.max(signal),
                    np.ptp(signal),  # peak-to-peak
                    self._skewness(signal),
                    self._kurtosis(signal)
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
        
        # Create feature names
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
        freqs = np.fft.fftfreq(n_times, 1/self.sfreq)
        freqs = freqs[:n_times//2]  # Only positive frequencies
        
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
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                sample_features.append(spectral_centroid)
                
                # Spectral rolloff (85% of power)
                cumsum_power = np.cumsum(power_spectrum)
                rolloff_idx = np.where(cumsum_power >= 0.85 * cumsum_power[-1])[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                sample_features.append(spectral_rolloff)
                
                # Spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
                sample_features.append(spectral_bandwidth)
                
                # Peak frequency
                peak_freq_idx = np.argmax(power_spectrum)
                peak_frequency = freqs[peak_freq_idx]
                sample_features.append(peak_frequency)
            
            features.append(sample_features)
        
        # Create feature names
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
                
                # Autocorrelation features
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # First zero crossing of autocorrelation
                zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                if len(zero_crossings) > 0:
                    first_zero = zero_crossings[0]
                    sample_features.append(first_zero)
                else:
                    sample_features.append(len(autocorr))
                
                # Autocorrelation decay
                decay = np.sum(autocorr[1:10])  # Sum of first 9 lags
                sample_features.append(decay)
                
                # Trend features
                x = np.arange(len(signal))
                slope, intercept, r_value, p_value, std_err = self._linear_regression(x, signal)
                sample_features.extend([slope, r_value, p_value])
                
                # Zero crossing rate
                zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
                sample_features.append(zcr)
                
                # Mean crossing rate
                mean_val = np.mean(signal)
                mcr = np.sum(np.diff(np.sign(signal - mean_val)) != 0) / len(signal)
                sample_features.append(mcr)
            
            features.append(sample_features)
        
        # Create feature names
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
        
        # Create feature names
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
                
                # Basic catch22-like features
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
                
                # Shannon entropy
                hist, _ = np.histogram(signal, bins=50)
                hist = hist / np.sum(hist)
                shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))
                sample_features.append(shannon_entropy)
                
                # Approximate entropy (simplified)
                approx_entropy = self._approximate_entropy(signal, m=2, r=0.2)
                sample_features.append(approx_entropy)
                
                # Sample entropy (simplified)
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

def create_synthetic_eeg_data(n_samples=1000, n_chans=129, n_times=200):
    """Create synthetic EEG data for demonstration"""
    logger.info(f"Creating synthetic EEG data: {n_samples} samples, {n_chans} channels, {n_times} time points")
    
    # Generate synthetic EEG-like data
    eeg_data = np.random.randn(n_samples, n_chans, n_times) * 0.1
    
    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add alpha rhythm (8-12 Hz)
        t = np.linspace(0, 2, n_times)  # 2 seconds
        alpha = 0.05 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
        eeg_data[i, :, :] += alpha
        
        # Add some channel-specific patterns
        for ch in range(n_chans):
            if ch % 10 == 0:  # Every 10th channel
                eeg_data[i, ch, :] += 0.02 * np.sin(2 * np.pi * 5 * t)
    
    # Generate response times with some correlation to the data
    alpha_power = np.mean(eeg_data[:, :, 50:150], axis=(1, 2))  # Alpha band power
    response_times = 0.5 + 0.3 * np.exp(-alpha_power) + np.random.normal(0, 0.1, n_samples)
    response_times = np.clip(response_times, 0.2, 2.0)  # Clip to realistic range
    
    return eeg_data, response_times

def run_feature_extraction():
    """Run the feature extraction pipeline"""
    logger.info("Starting feature extraction pipeline")
    start_time = datetime.now()
    
    # Create synthetic data (replace with real data loading)
    logger.info("Creating synthetic EEG data...")
    eeg_data, response_times = create_synthetic_eeg_data(n_samples=2000, n_chans=129, n_times=200)
    
    # Initialize feature extractor
    extractor = SimpleEEGFeatureExtractor(sfreq=100, n_chans=129, n_times=200)
    
    # Extract all features
    logger.info("Extracting comprehensive features...")
    features = extractor.extract_all_features(eeg_data)
    
    # Save features
    logger.info("Saving features...")
    np.save('eeg_features.npy', features)
    np.save('response_times.npy', response_times)
    
    with open('feature_names.json', 'w') as f:
        json.dump(extractor.feature_names, f, indent=2)
    
    # Basic analysis
    logger.info("Performing basic analysis...")
    
    # Check for NaN or infinite values
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    
    logger.info(f"NaN values: {nan_count}")
    logger.info(f"Infinite values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        logger.warning("Cleaning data...")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Feature statistics
    feature_stats = {
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0)
    }
    
    # Save statistics
    with open('feature_stats.json', 'w') as f:
        json.dump({k: v.tolist() for k, v in feature_stats.items()}, f, indent=2)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Feature importance (correlation with target)
    correlations = np.corrcoef(features.T, response_times)[-1, :-1]
    correlations = np.abs(correlations)
    
    # Top 20 most correlated features
    top_indices = np.argsort(correlations)[-20:]
    top_correlations = correlations[top_indices]
    top_names = [extractor.feature_names[i] for i in top_indices]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_correlations)), top_correlations)
    plt.yticks(range(len(top_correlations)), top_names)
    plt.xlabel('Absolute Correlation with Response Time')
    plt.title('Top 20 Most Correlated Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature distributions
    plt.figure(figsize=(15, 10))
    for i in range(min(20, features.shape[1])):
        plt.subplot(4, 5, i + 1)
        plt.hist(features[:, i], bins=50, alpha=0.7)
        plt.title(f'{extractor.feature_names[i][:20]}...')
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
        'n_samples': len(response_times),
        'n_features': features.shape[1],
        'feature_categories': {
            'statistical': True,
            'spectral': True,
            'temporal': True,
            'channel_specific': True,
            'catch22_like': True,
            'entropy': True,
            'connectivity': True
        },
        'data_quality': {
            'nan_values': int(nan_count),
            'infinite_values': int(inf_count)
        }
    }
    
    with open('feature_extraction_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Feature extraction complete! Duration: {duration}")
    logger.info(f"Total features: {features.shape[1]}")
    logger.info(f"Number of samples: {len(response_times)}")
    
    return features, response_times, extractor.feature_names

def main():
    """Main function to run the feature extraction pipeline"""
    logger.info("Starting Simple EEG Feature Extraction Pipeline")
    
    try:
        # Run the feature extraction
        features, response_times, feature_names = run_feature_extraction()
        
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
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise

if __name__ == "__main__":
    main()
