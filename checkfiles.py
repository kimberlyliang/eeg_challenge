import os
from pathlib import Path
from collections import defaultdict

root = Path("/users/kimliang/eeg_challenge/data_new_new")
task_name = "contrastChangeDetection"

subjects_with_bdf = []
subjects_without_bdf = []
subjects_missing_eeg_dir = []
subjects_by_release = defaultdict(lambda: {'with_bdf': [], 'without_bdf': []})

# Loop through release folders
for release in sorted(root.glob("release_*")):
    if not release.is_dir():
        continue

    release_name = release.name

    # Find ds*****-bdf folder(s)
    for ds in release.glob("*-bdf"):
        if not ds.is_dir():
            continue

        # Find subject folders
        for sub in ds.glob("sub-*"):
            eeg_dir = sub / "eeg"
            if not eeg_dir.exists():
                subjects_missing_eeg_dir.append(str(sub))
                subjects_by_release[release_name]['without_bdf'].append(str(sub))
                continue

            # Look for BDF files with contrastChangeDetection task
            bdf_files = list(eeg_dir.glob(f"*{task_name}*.bdf"))
            
            if bdf_files:
                subjects_with_bdf.append(str(sub))
                subjects_by_release[release_name]['with_bdf'].append(str(sub))
            else:
                subjects_without_bdf.append(str(sub))
                subjects_by_release[release_name]['without_bdf'].append(str(sub))

# ==== REPORT ====

print("\n" + "=" * 80)
print(f"BDF FILE CHECK FOR CONTRAST CHANGE DETECTION TASK")
print("=" * 80 + "\n")

print(f"📊 SUMMARY:")
print(f"   Subjects with BDF files: {len(subjects_with_bdf)}")
print(f"   Subjects missing BDF files: {len(subjects_without_bdf)}")
print(f"   Subjects missing eeg/ directory: {len(subjects_missing_eeg_dir)}")
print(f"   Total subjects checked: {len(subjects_with_bdf) + len(subjects_without_bdf) + len(subjects_missing_eeg_dir)}")

print(f"\n📋 BREAKDOWN BY RELEASE:")
print("-" * 80)
print(f"{'Release':<15} {'With BDF':<12} {'Missing BDF':<15} {'Total':<10}")
print("-" * 80)

for release in sorted(subjects_by_release.keys()):
    stats = subjects_by_release[release]
    with_count = len(stats['with_bdf'])
    without_count = len(stats['without_bdf'])
    total = with_count + without_count
    print(f"{release:<15} {with_count:<12} {without_count:<15} {total:<10}")

print("-" * 80)
total_with = sum(len(s['with_bdf']) for s in subjects_by_release.values())
total_without = sum(len(s['without_bdf']) for s in subjects_by_release.values())
total_all = total_with + total_without
print(f"{'TOTAL':<15} {total_with:<12} {total_without:<15} {total_all:<10}")

if subjects_without_bdf:
    print(f"\n⚠️  SUBJECTS MISSING BDF FILES (showing first 20):")
    for i, sub in enumerate(subjects_without_bdf[:20]):
        print(f"   - {sub}")
    if len(subjects_without_bdf) > 20:
        print(f"   ... and {len(subjects_without_bdf) - 20} more")

if subjects_missing_eeg_dir:
    print(f"\n⚠️  SUBJECTS MISSING EEG DIRECTORY (showing first 10):")
    for i, sub in enumerate(subjects_missing_eeg_dir[:10]):
        print(f"   - {sub}")
    if len(subjects_missing_eeg_dir) > 10:
        print(f"   ... and {len(subjects_missing_eeg_dir) - 10} more")

print("\n" + "=" * 80)
print("Done.\n")
