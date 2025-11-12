#!/usr/bin/env python3
"""
Minimal Feature Extraction for EEG Challenge 1

This script works with only Python standard library and basic numpy.
No external dependencies required!
"""

import math
import json
from datetime import datetime
import os

# Try to import numpy, fall back to basic math if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not available, using basic math functions")

def create_synthetic_data(n_samples=1000, n_chans=129, n_times=200):
    """Create synthetic EEG data using only basic Python"""
    print(f"Creating synthetic EEG data: {n_samples} samples, {n_chans} channels, {n_times} time points")
    
    if HAS_NUMPY:
        # Use numpy if available
        eeg_data = np.random.randn(n_samples, n_chans, n_times) * 0.1
        
        # Add alpha rhythm
        t = np.linspace(0, 2, n_times)
        alpha = 0.05 * np.sin(2 * np.pi * 10 * t)
        eeg_data += alpha[:, np.newaxis, :]
        
        # Generate response times
        alpha_power = np.mean(eeg_data[:, :, 50:150], axis=(1, 2))
        response_times = 0.5 + 0.3 * np.exp(-alpha_power) + np.random.normal(0, 0.1, n_samples)
        response_times = np.clip(response_times, 0.2, 2.0)
        
        return eeg_data, response_times
    else:
        # Use basic Python without numpy
        import random
        eeg_data = []
        response_times = []
        
        for i in range(n_samples):
            sample = []
            for ch in range(n_chans):
                channel = []
                for t in range(n_times):
                    # Generate synthetic EEG-like signal
                    time_val = t / 100.0  # 100 Hz sampling
                    alpha = 0.05 * math.sin(2 * math.pi * 10 * time_val)
                    noise = random.gauss(0, 0.1)
                    signal_val = alpha + noise
                    channel.append(signal_val)
                sample.append(channel)
            eeg_data.append(sample)
            
            # Generate response time
            alpha_power = sum(sum(sample[ch][50:150]) for ch in range(n_chans)) / (n_chans * 100)
            rt = 0.5 + 0.3 * math.exp(-alpha_power) + random.gauss(0, 0.1)
            rt = max(0.2, min(2.0, rt))  # Clip to realistic range
            response_times.append(rt)
        
        return eeg_data, response_times

def extract_basic_features(eeg_data, n_chans=129, n_times=200):
    """Extract basic features using only standard library"""
    print("Extracting basic features...")
    
    features = []
    feature_names = []
    
    # Create feature names first
    for ch in range(n_chans):
        feature_names.extend([
            f"ch{ch}_mean", f"ch{ch}_std", f"ch{ch}_min", f"ch{ch}_max",
            f"ch{ch}_range", f"ch{ch}_energy", f"ch{ch}_power"
        ])
    
    # Add global features
    feature_names.extend([
        "global_mean", "global_std", "global_min", "global_max", "global_range"
    ])
    
    for sample in eeg_data:
        sample_features = []
        
        # Channel-specific features
        for ch in range(n_chans):
            signal = sample[ch]
            
            # Basic statistics
            mean_val = sum(signal) / len(signal)
            variance = sum((x - mean_val) ** 2 for x in signal) / len(signal)
            std_val = math.sqrt(variance)
            min_val = min(signal)
            max_val = max(signal)
            range_val = max_val - min_val
            
            # Energy and power
            energy = sum(x ** 2 for x in signal)
            power = energy / len(signal)
            
            sample_features.extend([mean_val, std_val, min_val, max_val, range_val, energy, power])
        
        # Global features
        all_values = [val for channel in sample for val in channel]
        global_mean = sum(all_values) / len(all_values)
        global_variance = sum((x - global_mean) ** 2 for x in all_values) / len(all_values)
        global_std = math.sqrt(global_variance)
        global_min = min(all_values)
        global_max = max(all_values)
        global_range = global_max - global_min
        
        sample_features.extend([global_mean, global_std, global_min, global_max, global_range])
        
        features.append(sample_features)
    
    return features, feature_names

def extract_spectral_features(eeg_data, n_chans=129, n_times=200, sfreq=100):
    """Extract basic spectral features using FFT"""
    print("Extracting spectral features...")
    
    features = []
    feature_names = []
    
    # Frequency bands
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Create feature names
    for ch in range(n_chans):
        for band_name in freq_bands.keys():
            feature_names.append(f"ch{ch}_{band_name}_power")
        feature_names.extend([f"ch{ch}_peak_freq", f"ch{ch}_spectral_centroid"])
    
    for sample in eeg_data:
        sample_features = []
        
        for ch in range(n_chans):
            signal = sample[ch]
            
            # Simple FFT using basic math
            n = len(signal)
            freqs = [i * sfreq / n for i in range(n // 2)]
            
            # Calculate power spectrum
            power_spectrum = []
            for k in range(n // 2):
                real_part = sum(signal[i] * math.cos(2 * math.pi * k * i / n) for i in range(n))
                imag_part = sum(signal[i] * math.sin(2 * math.pi * k * i / n) for i in range(n))
                power = (real_part ** 2 + imag_part ** 2) / n
                power_spectrum.append(power)
            
            # Band power features
            for band_name, (low, high) in freq_bands.items():
                band_power = 0
                for i, freq in enumerate(freqs):
                    if low <= freq <= high:
                        band_power += power_spectrum[i]
                sample_features.append(band_power)
            
            # Peak frequency
            peak_idx = power_spectrum.index(max(power_spectrum))
            peak_freq = freqs[peak_idx]
            sample_features.append(peak_freq)
            
            # Spectral centroid
            total_power = sum(power_spectrum)
            if total_power > 0:
                centroid = sum(freq * power for freq, power in zip(freqs, power_spectrum)) / total_power
            else:
                centroid = 0
            sample_features.append(centroid)
        
        features.append(sample_features)
    
    return features, feature_names

def extract_temporal_features(eeg_data, n_chans=129, n_times=200):
    """Extract temporal features"""
    print("Extracting temporal features...")
    
    features = []
    feature_names = []
    
    # Create feature names
    for ch in range(n_chans):
        feature_names.extend([
            f"ch{ch}_zcr", f"ch{ch}_mcr", f"ch{ch}_autocorr_lag1",
            f"ch{ch}_autocorr_lag2", f"ch{ch}_trend_slope"
        ])
    
    for sample in eeg_data:
        sample_features = []
        
        for ch in range(n_chans):
            signal = sample[ch]
            
            # Zero crossing rate
            zcr = sum(1 for i in range(1, len(signal)) if (signal[i] >= 0) != (signal[i-1] >= 0)) / len(signal)
            sample_features.append(zcr)
            
            # Mean crossing rate
            mean_val = sum(signal) / len(signal)
            mcr = sum(1 for i in range(1, len(signal)) if (signal[i] >= mean_val) != (signal[i-1] >= mean_val)) / len(signal)
            sample_features.append(mcr)
            
            # Autocorrelation (simplified)
            def autocorr(signal, lag):
                if lag >= len(signal):
                    return 0
                n = len(signal) - lag
                mean_val = sum(signal) / len(signal)
                numerator = sum((signal[i] - mean_val) * (signal[i + lag] - mean_val) for i in range(n))
                denominator = sum((signal[i] - mean_val) ** 2 for i in range(len(signal)))
                return numerator / denominator if denominator != 0 else 0
            
            sample_features.append(autocorr(signal, 1))
            sample_features.append(autocorr(signal, 2))
            
            # Trend slope (linear regression)
            n = len(signal)
            x = list(range(n))
            sum_x = sum(x)
            sum_y = sum(signal)
            sum_xy = sum(x[i] * signal[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            else:
                slope = 0
            sample_features.append(slope)
        
        features.append(sample_features)
    
    return features, feature_names

def save_features(features, response_times, feature_names, filename_prefix="minimal"):
    """Save features to files"""
    print("Saving features...")
    
    # Save as JSON (works without numpy)
    with open(f"{filename_prefix}_features.json", 'w') as f:
        json.dump(features, f, indent=2)
    
    with open(f"{filename_prefix}_response_times.json", 'w') as f:
        json.dump(response_times, f, indent=2)
    
    with open(f"{filename_prefix}_feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Try to save as numpy if available
    if HAS_NUMPY:
        np.save(f"{filename_prefix}_features.npy", np.array(features))
        np.save(f"{filename_prefix}_response_times.npy", np.array(response_times))
    
    print(f"Features saved to {filename_prefix}_* files")

def create_simple_plots(features, response_times, feature_names):
    """Create simple plots using basic Python"""
    print("Creating simple plots...")
    
    # Find most correlated features
    correlations = []
    for i, feature in enumerate(features[0]):  # Use first sample to get feature count
        if i < len(feature_names):
            # Calculate correlation (simplified)
            feature_values = [sample[i] for sample in features]
            n = len(feature_values)
            mean_feature = sum(feature_values) / n
            mean_target = sum(response_times) / n
            
            numerator = sum((feature_values[j] - mean_feature) * (response_times[j] - mean_target) for j in range(n))
            denominator = math.sqrt(sum((feature_values[j] - mean_feature) ** 2 for j in range(n)) * 
                                  sum((response_times[j] - mean_target) ** 2 for j in range(n)))
            
            if denominator != 0:
                corr = abs(numerator / denominator)
            else:
                corr = 0
            
            correlations.append((corr, feature_names[i]))
    
    # Sort by correlation
    correlations.sort(reverse=True)
    
    # Save top correlations
    with open("feature_correlations.json", 'w') as f:
        json.dump(correlations[:20], f, indent=2)
    
    print(f"✅ Top 20 most correlated features saved to feature_correlations.json")
    print("Top 5 most correlated features:")
    for i, (corr, name) in enumerate(correlations[:5]):
        print(f"  {i+1}. {name}: {corr:.4f}")

def main():
    """Main function"""
    print("🧠 EEG Challenge 1: Minimal Feature Extraction")
    print("=" * 50)
    print("This script works with only Python standard library!")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Create synthetic data
    eeg_data, response_times = create_synthetic_data(n_samples=1000, n_chans=129, n_times=200)
    
    # Extract features
    all_features = []
    all_feature_names = []
    
    # Basic features
    basic_features, basic_names = extract_basic_features(eeg_data)
    all_features.extend(basic_features)
    all_feature_names.extend(basic_names)
    
    # Spectral features
    spectral_features, spectral_names = extract_spectral_features(eeg_data)
    all_features.extend(spectral_features)
    all_feature_names.extend(spectral_names)
    
    # Temporal features
    temporal_features, temporal_names = extract_temporal_features(eeg_data)
    all_features.extend(temporal_features)
    all_feature_names.extend(temporal_names)
    
    # Save features
    save_features(all_features, response_times, all_feature_names)
    
    # Create plots
    create_simple_plots(all_features, response_times, all_feature_names)
    
    # Generate report
    end_time = datetime.now()
    duration = end_time - start_time
    
    report = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "n_samples": len(response_times),
        "n_features": len(all_feature_names),
        "feature_categories": {
            "basic": len(basic_names),
            "spectral": len(spectral_names),
            "temporal": len(temporal_names)
        },
        "has_numpy": HAS_NUMPY
    }
    
    with open("minimal_extraction_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("MINIMAL FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Total features extracted: {len(all_feature_names)}")
    print(f"Number of samples: {len(response_times)}")
    print(f"Duration: {duration}")
    print("\nFiles saved:")
    print("- minimal_features.json: All extracted features")
    print("- minimal_response_times.json: Target response times")
    print("- minimal_feature_names.json: Names of all features")
    print("- feature_correlations.json: Feature correlations")
    print("- minimal_extraction_report.json: Summary report")
    
    if HAS_NUMPY:
        print("- minimal_features.npy: Features as numpy array")
        print("- minimal_response_times.npy: Response times as numpy array")

if __name__ == "__main__":
    main()
