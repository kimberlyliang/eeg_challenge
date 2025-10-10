from pathlib import Path
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Helper: download one release
def download_release(release_id: int):
    print(f"=== Starting download for Release R{release_id} ===")
    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=f"R{release_id}",
        cache_dir=DATA_DIR,
        mini=False,   # <-- FULL DATASET
    )

    raws = Parallel(n_jobs=-1)(
        delayed(lambda d: d.raw)(d) for d in dataset.datasets
    )

    print(f"✅ Finished Release R{release_id} — {len(dataset.datasets)} recordings downloaded.")
    return len(dataset.datasets)

total = 0
for i in range(1, 12):
    try:
        total += download_release(i)
    except Exception as e:
        print(f"Skipped Release R{i} due to error: {e}")

print(f"Done: Data stored in: {DATA_DIR.resolve()}")
