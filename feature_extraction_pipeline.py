#!/usr/bin/env python3
"""
Comprehensive Feature Extraction Pipeline for EEG Challenge 1

This script extracts catch22 features and other important EEG features
that can run overnight for comprehensive analysis.

Features included:
- catch22 features (22 time series features)
- Spectral features (power, coherence, etc.)
- Statistical features
- Wavelet features
- Entropy features
- Connectivity features
- Channel-specific features
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Feature extraction libraries
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

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    print("pywt not available. Install with: pip install PyWavelets")
    WAVELET_AVAILABLE = False

try:
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    print("sklearn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

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

class EEGFeatureExtractor:
    """Comprehensive EEG feature extraction class"""
    
    def __init__(self, sfreq=100, n_chans=129, n_times=200):
        self.sfreq = sfreq
        self.n_chans = n_chans
        self.n_times = n_times
        self.feature_names = []
        self.feature_importance = {}
        
        # Frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        logger.info(f"Initialized EEGFeatureExtractor with {n_chans} channels, {n_times} time points")
    
    def extract_all_features(self, eeg_data: np.ndarray, parallel=True) -> np.ndarray:
        """
        Extract all features from EEG data
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_chans, n_times)
            parallel: Whether to use parallel processing
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        logger.info(f"Extracting features from {eeg_data.shape[0]} samples")
        
        all_features = []
        feature_names = []
        
        # 1. Catch22 features
        if CATCH22_AVAILABLE:
            logger.info("Extracting catch22 features...")
            catch22_features, catch22_names = self._extract_catch22_features(eeg_data, parallel)
            all_features.append(catch22_features)
            feature_names.extend(catch22_names)
        
        # 2. Spectral features
        if SCIPY_AVAILABLE:
            logger.info("Extracting spectral features...")
            spectral_features, spectral_names = self._extract_spectral_features(eeg_data)
            all_features.append(spectral_features)
            feature_names.extend(spectral_names)
        
        # 3. Statistical features
        logger.info("Extracting statistical features...")
        stat_features, stat_names = self._extract_statistical_features(eeg_data)
        all_features.append(stat_features)
        feature_names.extend(stat_names)
        
        # 4. Wavelet features
        if WAVELET_AVAILABLE:
            logger.info("Extracting wavelet features...")
            wavelet_features, wavelet_names = self._extract_wavelet_features(eeg_data)
            all_features.append(wavelet_features)
            feature_names.extend(wavelet_names)
        
        # 5. Entropy features
        if SCIPY_AVAILABLE:
            logger.info("Extracting entropy features...")
            entropy_features, entropy_names = self._extract_entropy_features(eeg_data)
            all_features.append(entropy_features)
            feature_names.extend(entropy_names)
        
        # 6. Connectivity features
        if SCIPY_AVAILABLE:
            logger.info("Extracting connectivity features...")
            conn_features, conn_names = self._extract_connectivity_features(eeg_data)
            all_features.append(conn_features)
            feature_names.extend(conn_names)
        
        # 7. Channel-specific features
        logger.info("Extracting channel-specific features...")
        channel_features, channel_names = self._extract_channel_features(eeg_data)
        all_features.append(channel_features)
        feature_names.extend(channel_names)
        
        # 8. Temporal features
        logger.info("Extracting temporal features...")
        temporal_features, temporal_names = self._extract_temporal_features(eeg_data)
        all_features.append(temporal_features)
        feature_names.extend(temporal_names)
        
        # Combine all features
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)
            self.feature_names = feature_names
            logger.info(f"Extracted {combined_features.shape[1]} features total")
            return combined_features
        else:
            logger.error("No features extracted!")
            return np.array([])
    
    def _extract_catch22_features(self, eeg_data: np.ndarray, parallel=True) -> Tuple[np.ndarray, List[str]]:
        """Extract catch22 features"""
        n_samples, n_chans, n_times = eeg_data.shape
        
        if not CATCH22_AVAILABLE:
            return np.array([]), []
        
        # Flatten data for catch22 (it expects 1D time series)
        flattened_data = eeg_data.reshape(-1, n_times)
        
        if parallel:
            # Parallel processing
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(tqdm(
                    pool.imap(self._extract_catch22_single, flattened_data),
                    total=len(flattened_data),
                    desc="Catch22 features"
                ))
        else:
            # Sequential processing
            results = []
            for i in tqdm(range(len(flattened_data)), desc="Catch22 features"):
                results.append(self._extract_catch22_single(flattened_data[i]))
        
        # Reshape results
        catch22_features = np.array(results).reshape(n_samples, n_chans, -1)
        
        # Average across channels for each feature
        catch22_avg = np.mean(catch22_features, axis=1)
        
        # Create feature names
        feature_names = [f"catch22_{i}" for i in range(catch22_avg.shape[1])]
        
        return catch22_avg, feature_names
    
    def _extract_catch22_single(self, time_series: np.ndarray) -> np.ndarray:
        """Extract catch22 features for a single time series"""
        try:
            result = catch22_all(time_series)
            return np.array([result['values'][key] for key in result['names']])
        except:
            return np.zeros(22)  # Return zeros if extraction fails
    
    def _extract_spectral_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                # Power spectral density
                freqs, psd = welch(eeg_data[i, ch, :], fs=self.sfreq, nperseg=min(64, n_times//2))
                
                # Band power features
                for band_name, (low, high) in self.freq_bands.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_power = np.sum(psd[band_mask])
                    sample_features.append(band_power)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                sample_features.append(spectral_centroid)
                
                # Spectral rolloff
                cumsum_psd = np.cumsum(psd)
                rolloff_idx = np.where(cumsum_psd >= 0.85 * cumsum_psd[-1])[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                sample_features.append(spectral_rolloff)
                
                # Spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
                sample_features.append(spectral_bandwidth)
            
            features.append(sample_features)
        
        # Create feature names
        for ch in range(n_chans):
            for band_name in self.freq_bands.keys():
                feature_names.append(f"ch{ch}_{band_name}_power")
            feature_names.extend([f"ch{ch}_spectral_centroid", f"ch{ch}_spectral_rolloff", f"ch{ch}_spectral_bandwidth"])
        
        return np.array(features), feature_names
    
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
                    stats.skew(signal),
                    stats.kurtosis(signal)
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
    
    def _extract_wavelet_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract wavelet features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        # Wavelet decomposition
        wavelet = 'db4'
        levels = 4
        
        for i in range(n_samples):
            sample_features = []
            
            for ch in range(n_chans):
                signal = eeg_data[i, ch, :]
                
                try:
                    # Wavelet decomposition
                    coeffs = pywt.wavedec(signal, wavelet, level=levels)
                    
                    # Energy of each level
                    for level, coeff in enumerate(coeffs):
                        energy = np.sum(coeff ** 2)
                        sample_features.append(energy)
                    
                    # Relative energy
                    total_energy = sum(np.sum(c ** 2) for c in coeffs)
                    for level, coeff in enumerate(coeffs):
                        rel_energy = np.sum(coeff ** 2) / total_energy
                        sample_features.append(rel_energy)
                    
                    # Entropy of coefficients
                    for level, coeff in enumerate(coeffs):
                        if len(coeff) > 1:
                            coeff_entropy = entropy(np.abs(coeff) + 1e-10)
                            sample_features.append(coeff_entropy)
                        else:
                            sample_features.append(0)
                
                except:
                    # If wavelet decomposition fails, add zeros
                    sample_features.extend([0] * (levels + 1 + levels + levels))
            
            features.append(sample_features)
        
        # Create feature names
        for ch in range(n_chans):
            for level in range(levels + 1):
                feature_names.append(f"ch{ch}_wavelet_energy_L{level}")
            for level in range(levels + 1):
                feature_names.append(f"ch{ch}_wavelet_rel_energy_L{level}")
            for level in range(levels + 1):
                feature_names.append(f"ch{ch}_wavelet_entropy_L{level}")
        
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
                shannon_entropy = entropy(hist + 1e-10)
                sample_features.append(shannon_entropy)
                
                # Approximate entropy
                try:
                    approx_entropy = self._approximate_entropy(signal, m=2, r=0.2)
                    sample_features.append(approx_entropy)
                except:
                    sample_features.append(0)
                
                # Sample entropy
                try:
                    sample_entropy = self._sample_entropy(signal, m=2, r=0.2)
                    sample_features.append(sample_entropy)
                except:
                    sample_features.append(0)
            
            features.append(sample_features)
        
        # Create feature names
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_shannon_entropy", f"ch{ch}_approx_entropy", f"ch{ch}_sample_entropy"])
        
        return np.array(features), feature_names
    
    def _approximate_entropy(self, data, m, r):
        """Calculate approximate entropy"""
        N = len(data)
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
            phi = np.mean(np.log(C))
            return phi
        
        return _approximate_entropy_single(data, m, r) - _approximate_entropy_single(data, m + 1, r)
    
    def _sample_entropy(self, data, m, r):
        """Calculate sample entropy"""
        N = len(data)
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
            return -np.log(A / B)
        
        return _sample_entropy_single(data, m, r)
    
    def _extract_connectivity_features(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Extract connectivity features"""
        n_samples, n_chans, n_times = eeg_data.shape
        features = []
        feature_names = []
        
        # Select subset of channels for connectivity (to avoid memory issues)
        n_conn_chans = min(20, n_chans)
        conn_indices = np.linspace(0, n_chans-1, n_conn_chans, dtype=int)
        
        for i in range(n_samples):
            sample_features = []
            
            # Calculate connectivity between selected channels
            for ch1 in range(n_conn_chans):
                for ch2 in range(ch1 + 1, n_conn_chans):
                    idx1, idx2 = conn_indices[ch1], conn_indices[ch2]
                    
                    # Coherence
                    f, coh = coherence(eeg_data[i, idx1, :], eeg_data[i, idx2, :], fs=self.sfreq)
                    mean_coh = np.mean(coh)
                    sample_features.append(mean_coh)
                    
                    # Cross-correlation
                    corr = np.corrcoef(eeg_data[i, idx1, :], eeg_data[i, idx2, :])[0, 1]
                    sample_features.append(corr)
                    
                    # Phase locking value (simplified)
                    phase1 = np.angle(signal.hilbert(eeg_data[i, idx1, :]))
                    phase2 = np.angle(signal.hilbert(eeg_data[i, idx2, :]))
                    plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
                    sample_features.append(plv)
            
            features.append(sample_features)
        
        # Create feature names
        for ch1 in range(n_conn_chans):
            for ch2 in range(ch1 + 1, n_conn_chans):
                idx1, idx2 = conn_indices[ch1], conn_indices[ch2]
                feature_names.extend([f"coh_ch{idx1}_ch{idx2}", f"corr_ch{idx1}_ch{idx2}", f"plv_ch{idx1}_ch{idx2}"])
        
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
                
                # Zero crossing rate
                zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
                sample_features.append(zcr)
                
                # Mean crossing rate
                mean_val = np.mean(signal)
                mcr = np.sum(np.diff(np.sign(signal - mean_val)) != 0) / len(signal)
                sample_features.append(mcr)
                
                # Signal energy
                energy = np.sum(signal ** 2)
                sample_features.append(energy)
                
                # Signal power
                power = np.mean(signal ** 2)
                sample_features.append(power)
            
            features.append(sample_features)
        
        # Create feature names
        feature_names.extend(['global_mean', 'global_std', 'global_var', 'global_median', 'global_ptp'])
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_zcr", f"ch{ch}_mcr", f"ch{ch}_energy", f"ch{ch}_power"])
        
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
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, signal)
                sample_features.extend([slope, r_value, p_value])
            
            features.append(sample_features)
        
        # Create feature names
        for ch in range(n_chans):
            feature_names.extend([f"ch{ch}_autocorr_zero", f"ch{ch}_autocorr_decay", 
                                f"ch{ch}_trend_slope", f"ch{ch}_trend_r", f"ch{ch}_trend_p"])
        
        return np.array(features), feature_names

class FeatureAnalysis:
    """Feature analysis and selection"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, feature_names: List[str]):
        self.features = features
        self.targets = targets
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
    
    def analyze_features(self):
        """Comprehensive feature analysis"""
        logger.info("Starting feature analysis...")
        
        # Basic statistics
        logger.info(f"Feature matrix shape: {self.features.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        # Check for NaN values
        nan_count = np.isnan(self.features).sum()
        logger.info(f"NaN values in features: {nan_count}")
        
        if nan_count > 0:
            logger.warning("Replacing NaN values with 0")
            self.features = np.nan_to_num(self.features, nan=0.0)
        
        # Check for infinite values
        inf_count = np.isinf(self.features).sum()
        logger.info(f"Infinite values in features: {inf_count}")
        
        if inf_count > 0:
            logger.warning("Replacing infinite values with 0")
            self.features = np.nan_to_num(self.features, posinf=0.0, neginf=0.0)
        
        # Feature statistics
        feature_stats = {
            'mean': np.mean(self.features, axis=0),
            'std': np.std(self.features, axis=0),
            'min': np.min(self.features, axis=0),
            'max': np.max(self.features, axis=0)
        }
        
        # Save feature statistics
        with open('feature_stats.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in feature_stats.items()}, f, indent=2)
        
        logger.info("Feature analysis complete")
        return feature_stats
    
    def select_features(self, k=1000, method='f_regression'):
        """Select most important features"""
        logger.info(f"Selecting top {k} features using {method}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(self.features)
        
        # Feature selection
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, features_scaled.shape[1]))
            features_selected = selector.fit_transform(features_scaled, self.targets)
            
            # Get selected feature names and importance scores
            selected_indices = selector.get_support(indices=True)
            self.selected_features = features_selected
            self.feature_importance = selector.scores_[selected_indices]
            
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_indices)} features")
            
            return features_selected, selected_feature_names, self.feature_importance
        
        return features_scaled, self.feature_names, None
    
    def plot_feature_importance(self, top_n=50, save_path='feature_importance.png'):
        """Plot feature importance"""
        if self.feature_importance is None:
            logger.warning("No feature importance scores available")
            return
        
        # Get top N features
        top_indices = np.argsort(self.feature_importance)[-top_n:]
        top_scores = self.feature_importance[top_indices]
        top_names = [self.feature_names[i] for i in top_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_scores)), top_scores)
        plt.yticks(range(len(top_scores)), top_names)
        plt.xlabel('Feature Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Feature importance plot saved to {save_path}")
    
    def plot_feature_distributions(self, n_features=20, save_path='feature_distributions.png'):
        """Plot feature distributions"""
        # Select random features for visualization
        n_features = min(n_features, self.features.shape[1])
        feature_indices = np.random.choice(self.features.shape[1], n_features, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, idx in enumerate(feature_indices):
            if i < len(axes):
                axes[i].hist(self.features[:, idx], bins=50, alpha=0.7)
                axes[i].set_title(f'{self.feature_names[idx][:30]}...')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Feature distributions plot saved to {save_path}")

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

def run_overnight_feature_extraction():
    """Run comprehensive feature extraction overnight"""
    logger.info("Starting overnight feature extraction pipeline")
    start_time = datetime.now()
    
    # Create synthetic data (replace with real data loading)
    logger.info("Creating synthetic EEG data...")
    eeg_data, response_times = create_synthetic_eeg_data(n_samples=5000, n_chans=129, n_times=200)
    
    # Initialize feature extractor
    extractor = EEGFeatureExtractor(sfreq=100, n_chans=129, n_times=200)
    
    # Extract all features
    logger.info("Extracting comprehensive features...")
    features = extractor.extract_all_features(eeg_data, parallel=True)
    
    # Save raw features
    logger.info("Saving raw features...")
    np.save('raw_features.npy', features)
    np.save('response_times.npy', response_times)
    
    with open('feature_names.json', 'w') as f:
        json.dump(extractor.feature_names, f, indent=2)
    
    # Feature analysis
    logger.info("Performing feature analysis...")
    analyzer = FeatureAnalysis(features, response_times, extractor.feature_names)
    feature_stats = analyzer.analyze_features()
    
    # Feature selection
    logger.info("Selecting important features...")
    selected_features, selected_names, importance_scores = analyzer.select_features(k=1000)
    
    # Save selected features
    logger.info("Saving selected features...")
    np.save('selected_features.npy', selected_features)
    
    with open('selected_feature_names.json', 'w') as f:
        json.dump(selected_names, f, indent=2)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    analyzer.plot_feature_importance(top_n=50)
    analyzer.plot_feature_distributions(n_features=20)
    
    # Generate summary report
    end_time = datetime.now()
    duration = end_time - start_time
    
    report = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_hours': duration.total_seconds() / 3600,
        'n_samples': len(response_times),
        'n_features_total': features.shape[1],
        'n_features_selected': selected_features.shape[1],
        'feature_categories': {
            'catch22': CATCH22_AVAILABLE,
            'spectral': SCIPY_AVAILABLE,
            'wavelet': WAVELET_AVAILABLE,
            'entropy': SCIPY_AVAILABLE,
            'connectivity': SCIPY_AVAILABLE
        }
    }
    
    with open('feature_extraction_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Feature extraction complete! Duration: {duration}")
    logger.info(f"Total features: {features.shape[1]}")
    logger.info(f"Selected features: {selected_features.shape[1]}")
    
    return features, selected_features, response_times, extractor.feature_names, selected_names

def main():
    """Main function to run the feature extraction pipeline"""
    logger.info("Starting EEG Feature Extraction Pipeline")
    
    try:
        # Run the overnight feature extraction
        features, selected_features, response_times, feature_names, selected_names = run_overnight_feature_extraction()
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION COMPLETE!")
        print("="*60)
        print(f"Total features extracted: {features.shape[1]}")
        print(f"Selected features: {selected_features.shape[1]}")
        print(f"Number of samples: {len(response_times)}")
        print("\nFiles saved:")
        print("- raw_features.npy: All extracted features")
        print("- selected_features.npy: Selected important features")
        print("- response_times.npy: Target response times")
        print("- feature_names.json: Names of all features")
        print("- selected_feature_names.json: Names of selected features")
        print("- feature_extraction_report.json: Summary report")
        print("- feature_importance.png: Feature importance plot")
        print("- feature_distributions.png: Feature distribution plots")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise

if __name__ == "__main__":
    main()
