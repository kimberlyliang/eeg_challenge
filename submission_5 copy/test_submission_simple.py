"""
Simple test script to verify your submission works correctly.
This tests the submission file without requiring the full dataset.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add submission directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_submission():
    """Test the submission file can be imported and models can make predictions"""
    
    print("=" * 60)
    print("Testing Submission File")
    print("=" * 60)
    
    # Test 1: Import submission
    print("\n1. Testing import...")
    try:
        from submission import Submission, RandomForestWrapper, extract_features_from_window
        print("   ✅ Successfully imported Submission class")
    except Exception as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    # Test 2: Initialize submission
    print("\n2. Testing initialization...")
    try:
        SFREQ = 100
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sub = Submission(SFREQ, DEVICE)
        print(f"   ✅ Successfully initialized with SFREQ={SFREQ}, DEVICE={DEVICE}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 3: Load Challenge 1 model (Random Forest)
    print("\n3. Testing Challenge 1 model loading (Random Forest)...")
    try:
        # Debug: Check if file exists before loading
        from pathlib import Path
        test_file = Path(__file__).parent / "RandomForest_200_20251101_170036.pt"
        print(f"   Checking for model file...")
        print(f"   Expected location: {test_file}")
        print(f"   File exists: {test_file.exists()}")
        if test_file.exists():
            print(f"   File size: {test_file.stat().st_size / (1024*1024):.1f} MB")
        
        model_1 = sub.get_model_challenge_1()
        print(f"   ✅ Challenge 1 model loaded successfully")
        print(f"   Model type: {type(model_1)}")
    except Exception as e:
        print(f"   ❌ Failed to load Challenge 1 model: {e}")
        print(f"   Make sure RandomForest_200_20251101_170036.pt is in submission_5/ directory")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test feature extraction
    print("\n4. Testing feature extraction...")
    try:
        # Create synthetic EEG window: (n_chans=129, n_times=200)
        synthetic_window = np.random.randn(129, 200)
        features = extract_features_from_window(synthetic_window, fs=SFREQ)
        assert features.shape == (1161,), f"Expected 1161 features, got {features.shape}"
        print(f"   ✅ Feature extraction works correctly (extracted {len(features)} features)")
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return False
    
    # Test 5: Test model prediction on synthetic data
    print("\n5. Testing model prediction...")
    try:
        model_1.eval()
        
        # Create batch of synthetic EEG data: (batch_size=2, n_chans=129, n_times=200)
        batch_size = 2
        X_test = torch.randn(batch_size, 129, 200).float().to(DEVICE)
        
        with torch.inference_mode():
            y_pred = model_1.forward(X_test)
        
        # Check output shape
        assert y_pred.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {y_pred.shape}"
        assert not torch.isnan(y_pred).any(), "Predictions contain NaN values"
        assert not torch.isinf(y_pred).any(), "Predictions contain Inf values"
        
        print(f"   ✅ Model prediction works correctly")
        print(f"   Input shape: {X_test.shape}")
        print(f"   Output shape: {y_pred.shape}")
        print(f"   Sample predictions: {y_pred.squeeze().cpu().numpy()}")
    except Exception as e:
        print(f"   ❌ Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test Challenge 2 model loading (neural network)
    print("\n6. Testing Challenge 2 model loading (Neural Network)...")
    try:
        model_2 = sub.get_model_challenge_2()
        model_2.eval()
        print(f"   ✅ Challenge 2 model loaded successfully")
        
        # Quick prediction test
        X_test_2 = torch.randn(1, 129, 200).float().to(DEVICE)
        with torch.inference_mode():
            y_pred_2 = model_2.forward(X_test_2)
        assert y_pred_2.shape == (1, 1), f"Expected shape (1, 1), got {y_pred_2.shape}"
        print(f"   ✅ Challenge 2 model prediction works")
    except Exception as e:
        print(f"   ⚠️ Challenge 2 model issue (may not affect Challenge 1): {e}")
    
    # Test 7: Test with different input shapes (edge cases)
    print("\n7. Testing edge cases (different input shapes)...")
    try:
        model_1.eval()
        
        # Test single sample
        X_single = torch.randn(1, 129, 200).float().to(DEVICE)
        y_single = model_1.forward(X_single)
        assert y_single.shape == (1, 1), "Single sample failed"
        
        # Test larger batch
        X_large = torch.randn(8, 129, 200).float().to(DEVICE)
        y_large = model_1.forward(X_large)
        assert y_large.shape == (8, 1), "Large batch failed"
        
        print(f"   ✅ Edge cases handled correctly")
    except Exception as e:
        print(f"   ❌ Edge case failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your submission looks good.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Make sure all required model files are in submission_5/ directory:")
    print("   - RandomForest_200_20251101_170036.pt (or .joblib)")
    print("   - weights_challenge_2_10_30.pt")
    print("\n2. Test with actual data using local_scoring.py:")
    print("   python local_scoring.py --submission-zip submission_5.zip --data-dir data --fast-dev-run")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_submission()
    sys.exit(0 if success else 1)
