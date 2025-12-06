import os
from pathlib import Path
from collections import defaultdict

root = Path("/users/kimliang/eeg_challenge/data_new_new")

required_suffixes = [
    "_channels.tsv",
    "_eeg.bdf",
    "_eeg.json",
    "_events.tsv"
]

missing_subjects = []
incomplete_subjects = {}
complete_subjects = []

# Loop through release folders
for release in sorted(root.glob("release_*")):
    if not release.is_dir():
        continue

    # Find ds*****-bdf folder(s)
    for ds in release.glob("*-bdf"):
        if not ds.is_dir():
            continue

        # Find subject folders
        for sub in ds.glob("sub-*"):
            eeg_dir = sub / "eeg"
            if not eeg_dir.exists():
                missing_subjects.append(str(sub))
                continue

            # Collect files inside eeg/
            files = list(eeg_dir.glob("*"))
            prefix_groups = defaultdict(list)

            # Group by prefix (everything before the suffix)
            for f in files:
                name = f.name

                # Identify which required suffix it matches
                for suf in required_suffixes:
                    if name.endswith(suf):
                        prefix = name[: -len(suf)]
                        prefix_groups[prefix].append(suf)
                        break

            # Check completeness
            missing_for_this_subject = {}
            for prefix, found_suffixes in prefix_groups.items():
                missing = [s for s in required_suffixes if s not in found_suffixes]
                if missing:
                    missing_for_this_subject[prefix] = missing

            # Also check for tasks missing entirely
            # (prefixes with <4 files)
            if missing_for_this_subject:
                incomplete_subjects[str(sub)] = missing_for_this_subject
            else:
                complete_subjects.append(str(sub))

# ==== REPORT ====

print("\n===== EEG FILE COMPLETENESS REPORT =====\n")

print(f"Subjects missing eeg folder: {len(missing_subjects)}")
for s in missing_subjects:
    print("  -", s)

print(f"\nSubjects with incomplete task/run data: {len(incomplete_subjects)}")
for sub, problems in incomplete_subjects.items():
    print(f"\n--- {sub} ---")
    for prefix, missing in problems.items():
        print(f"  Task/run: {prefix}")
        print(f"    Missing: {missing}")

print(f"\nSubjects with FULL EEG data present: {len(complete_subjects)}")
for s in complete_subjects:
    print("  +", s)

print("\nDone.\n")
