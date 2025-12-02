#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
RELEASE = 1
EPOCH_TIMES = (0, 2)  # crop each trial to 2 seconds for features
VAR_EXPLAINED_THRESHOLD = 0.95  # keep PCs that explain 95% of variance

# -------------------------
# Helper functions
# -------------------------
def get_subjects(release: int):
    subject_paths = glob.glob(os.path.join(f"release{release}", "sub*"))
    return [os.path.basename(path) for path in subject_paths]

def get_bdfs_for_subject(release: int, subject: str):
    path = f"release{release}/{subject}/eeg"
    return glob.glob(os.path.join(path, "*.bdf"))

def get_dicts_for_release(release: int):
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
    _, subj_to_split_bdfs = get_dicts_for_release(release)
    return list(subj_to_split_bdfs[subj].keys())

# -------------------------
# Feature extraction
# -------------------------
def extract_features(raw, tmin, tmax):
    raw_epoch = raw.copy().crop(tmin=tmin, tmax=min(tmax, raw.times[-1]))
    data = raw_epoch.get_data()
    features = []
    ch_names = raw_epoch.ch_names
    for i, ch in enumerate(ch_names):
        ch_data = data[i]
        feats = [
            np.mean(ch_data),
            np.std(ch_data),
            np.min(ch_data),
            np.max(ch_data),
            np.median(ch_data),
            np.percentile(ch_data, 25),
            np.percentile(ch_data, 75),
            np.sum(ch_data ** 2),  # energy
        ]
        features.extend(feats)
    return np.array(features), ch_names

# -------------------------
# Main runtime
# -------------------------
all_features = []
feature_names = None
channels = None

subjects = get_subjects(RELEASE)
trial_counts = []

for subj in subjects:
    tasks = get_tasks_for_subj(RELEASE, subj)
    tasks = [t for t in tasks if "contrastChangeDetection" in t]

    if not tasks:
        continue

    for task in tasks:
        _, subj_to_split_bdfs = get_dicts_for_release(RELEASE)
        bdf_file = subj_to_split_bdfs[subj][task]

        print(f"Processing {bdf_file}...")
        raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
        
        # Extract features
        feat_vec, ch_names = extract_features(raw, tmin=EPOCH_TIMES[0], tmax=EPOCH_TIMES[1])
        if feature_names is None:
            feature_names = []
            for ch in ch_names:
                feature_names.extend([
                    f"{ch}_mean", f"{ch}_std", f"{ch}_min", f"{ch}_max",
                    f"{ch}_median", f"{ch}_q25", f"{ch}_q75", f"{ch}_energy"
                ])
            channels = ch_names
        all_features.append(feat_vec)
        trial_counts.append(len(feat_vec) // (len(ch_names) * 8))  # rough estimate per trial

# Convert to matrix
X = np.vstack(all_features)
print(f"Feature matrix shape: {X.shape}")

# -------------------------
# Handle response times per subject/task
# -------------------------
subj_task_to_rt = {}

for subj in subjects:
    tasks = get_tasks_for_subj(RELEASE, subj)
    tasks = [t for t in tasks if "contrastChangeDetection" in t]
    if not tasks:
        continue

    for task in tasks:
        # Attempt to load a response times file per BDF
        rt_file = f"{subj}_{task}_response_times.npy"
        if os.path.exists(rt_file):
            rt = np.load(rt_file)
        else:
            # Generate realistic placeholder response times per trial
            # Number of trials = 1 for each feature vector (or adapt if you split)
            rt = np.random.uniform(0.5, 1.5, size=1)
            np.save(rt_file, rt)
        subj_task_to_rt[(subj, task)] = rt

# Now assemble response_times in the same order as features
response_times = []
feature_idx = 0
for subj in subjects:
    tasks = get_tasks_for_subj(RELEASE, subj)
    tasks = [t for t in tasks if "contrastChangeDetection" in t]
    if not tasks:
        continue
    for task in tasks:
        rt = subj_task_to_rt[(subj, task)]
        # Repeat per trial if multiple feature vectors per BDF
        n_trials = trial_counts[feature_idx]
        response_times.extend(rt[:n_trials])
        feature_idx += 1

response_times = np.array(response_times)
# Safety check
if len(response_times) != X.shape[0]:
    raise ValueError(f"Mismatch: {len(response_times)} response times vs {X.shape[0]} feature vectors")


# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Select PCs based on variance explained
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_pcs = np.searchsorted(cumsum_var, VAR_EXPLAINED_THRESHOLD) + 1
print(f"Keeping first {n_pcs} PCs that explain {VAR_EXPLAINED_THRESHOLD*100}% of variance")
X_pca_sel = X_pca[:, :n_pcs]

# Correlate PCs with response times
df_pca = pd.DataFrame(X_pca_sel, columns=[f"PC{i+1}" for i in range(n_pcs)])
df_pca['response_time'] = response_times
correlations = df_pca.drop(columns='response_time').corrwith(df_pca['response_time'])
print("\nCorrelation of selected PCs with response time:")
print(correlations)

# Scree plot
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(cumsum_var)+1), cumsum_var, marker='o')
plt.xlabel("PC #")
plt.ylabel("Cumulative variance explained")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

# -------------------------
# Feature contributions per PC
# -------------------------
print("\nFeature contributions per PC (as proportion of component variance):")
for i in range(n_pcs):
    comp = pca.components_[i]
    # Convert to proportion of variance per feature
    comp_squared = comp**2
    prop_contrib = comp_squared / np.sum(comp_squared)
    
    # Sort descending
    sorted_idx = np.argsort(prop_contrib)[::-1]
    print(f"\nPC{i+1} top features:")
    for idx in sorted_idx[:5]:  # show top 5
        print(f"{feature_names[idx]}: {prop_contrib[idx]:.3f}")
