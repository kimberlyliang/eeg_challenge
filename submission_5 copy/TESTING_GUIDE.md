# Testing Your Submission

This guide explains how to test your submission to make sure it works correctly before submitting.

## Quick Test (Recommended First Step)

Run the simple test script that verifies your submission works without requiring the full dataset:

```bash
# Make sure you're in the project root and have your environment activated
cd /Users/kimberly/Documents/ESE5380/eeg_challenge

# Activate your environment (if you have one)
# source eeg_env/bin/activate  # or whatever your environment is

# Run the simple test
cd submission_5
python test_submission_simple.py
```

This test will:
- ✅ Verify the submission file can be imported
- ✅ Check that models can be loaded
- ✅ Test feature extraction works
- ✅ Verify predictions work on synthetic data
- ✅ Check edge cases (different batch sizes)

**Expected output:** You should see "✅ All tests passed!" if everything works.

## Full Test with Actual Data

After the simple test passes, test with real data using the local scoring script:

### Step 1: Create a submission zip file

```bash
cd submission_5
# Make sure these files are in the directory:
# - submission.py
# - RandomForest_200_20251101_170036.pt (or .joblib)
# - weights_challenge_2_10_30.pt

# Create the zip (flat structure, NO subdirectory!)
zip submission_5.zip submission.py RandomForest_200_20251101_170036.pt weights_challenge_2_10_30.pt
```

**Important:** The zip file must have a flat structure (all files at the root level, not in a folder).

### Step 2: Run local scoring

```bash
# From the project root
python local_scoring.py \
  --submission-zip submission_5/submission_5.zip \
  --data-dir data \
  --fast-dev-run
```

The `--fast-dev-run` flag will:
- Only test on one subject (much faster)
- Verify your submission runs without errors
- Give you sample scores

**Note:** Fast dev run scores won't be representative, but they verify everything works.

### Step 3: Full test (optional, slower)

For a more representative test (but slower):

```bash
python local_scoring.py \
  --submission-zip submission_5/submission_5.zip \
  --data-dir data
```

## Troubleshooting

### If the simple test fails:

1. **Model file not found:**
   - Make sure `RandomForest_200_20251101_170036.pt` is in `submission_5/` directory
   - Or update the filename in `submission.py` to match your actual model file

2. **Import errors:**
   - Make sure you have all required packages installed (numpy, scipy, sklearn, torch, joblib)
   - Activate your Python environment if you have one

3. **Feature extraction errors:**
   - Check that scipy and numpy are properly installed
   - Verify your Python version is compatible

### If local scoring fails:

1. **Check the zip structure:**
   ```bash
   unzip -l submission_5/submission_5.zip
   ```
   Should show files directly (not in a subfolder):
   ```
   submission.py
   RandomForest_200_20251101_170036.pt
   weights_challenge_2_10_30.pt
   ```

2. **Check file paths:**
   - Make sure all model files are in the zip
   - Check that the paths in `submission.py` match the actual filenames

3. **Check data directory:**
   - Make sure `data/` directory exists with the required releases

## What to Check Before Submission

- [ ] Simple test passes (`test_submission_simple.py`)
- [ ] Fast dev run completes without errors
- [ ] All required model files are in the zip
- [ ] Zip file has flat structure (no subdirectories)
- [ ] File names in `submission.py` match actual files in zip

## Submission Format

Your submission zip should contain:
```
submission_5.zip
├── submission.py
├── RandomForest_200_20251101_170036.pt  (or .joblib)
└── weights_challenge_2_10_30.pt
```

**NO folders inside the zip!**
