#!/usr/bin/env python3
"""
Merge split EEG challenge data releases into a single merged directory.

This script finds all split releases (e.g., release_1, release_1 2, release_1 3)
and merges them into a single release_1 folder in data_merged/.
"""

import shutil
from pathlib import Path
from collections import defaultdict

def find_split_releases(data_dir):
    """
    Find all split releases in the data directory.
    
    Returns:
        dict: {release_num: [list of split release paths]}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    split_releases = defaultdict(list)
    
    # Find all release folders
    for item in data_path.iterdir():
        if not item.is_dir():
            continue
        
        name = item.name
        
        # Check if it's a release folder (release_X or release_X 2, etc.)
        if name.startswith("release_"):
            # Extract release number
            parts = name.split("_", 1)
            if len(parts) == 2:
                release_part = parts[1]
                # Check if it's just a number or has a suffix
                if release_part.isdigit():
                    release_num = int(release_part)
                    split_releases[release_num].append((name, item, 0))  # 0 = primary
                elif " " in release_part:
                    # It's a split (e.g., "1 2" or "1 3")
                    num_part = release_part.split()[0]
                    if num_part.isdigit():
                        release_num = int(num_part)
                        # Get split number (2, 3, etc.)
                        split_num = int(release_part.split()[1]) if release_part.split()[1].isdigit() else 0
                        split_releases[release_num].append((name, item, split_num))
    
    # Sort by split number (0 = primary, then 2, 3, etc.)
    for release_num in split_releases:
        split_releases[release_num].sort(key=lambda x: x[2])
    
    return split_releases

def get_dataset_folder_name(release_num):
    """Get the expected dataset folder name for a release number."""
    # Mapping based on observed pattern
    dataset_mapping = {
        1: "ds005505-bdf",
        2: "ds005506-bdf",
        3: "ds005507-bdf",
        4: "ds005508-bdf",
        5: "ds005509-bdf",
        6: "ds005510-bdf",
        7: "ds005511-bdf",
        8: "ds005512-bdf",
        9: "ds005513-bdf",
        10: "ds005514-bdf",
        11: "ds005515-bdf",
    }
    return dataset_mapping.get(release_num, f"ds00550{release_num}-bdf")

def check_release_exists(output_dir, release_num):
    """
    Check if a merged release already exists and has content.
    
    Args:
        output_dir: Output directory for merged data
        release_num: Release number
    
    Returns:
        tuple: (exists: bool, subject_count: int)
    """
    merged_release_dir = output_dir / f"release_{release_num}"
    dataset_folder = get_dataset_folder_name(release_num)
    merged_dataset_dir = merged_release_dir / dataset_folder
    
    if not merged_dataset_dir.exists():
        return False, 0
    
    # Count subjects in the merged directory
    subject_dirs = [d for d in merged_dataset_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("sub-")]
    
    return len(subject_dirs) > 0, len(subject_dirs)

def merge_subject_files(source_dir, dest_dir):
    """Merge files from source subject directory into destination."""
    # Ensure destination exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files, overwriting if they exist
    for item in source_dir.rglob("*"):
        if item.is_file():
            # Calculate relative path
            rel_path = item.relative_to(source_dir)
            dest_path = dest_dir / rel_path
            
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file (overwrite if exists)
            shutil.copy2(item, dest_path)

def merge_release(split_paths, output_dir, release_num, dry_run=False, force=False):
    """
    Merge all split releases into a single merged release.
    
    Args:
        split_paths: List of (name, path, split_num) tuples
        output_dir: Output directory for merged data
        release_num: Release number
        dry_run: If True, only print what would be done
        force: If True, re-merge even if release already exists
    """
    print(f"\n{'='*70}")
    print(f"Merging Release {release_num}")
    print(f"{'='*70}")
    
    # Check if release already exists
    exists, subject_count = check_release_exists(output_dir, release_num)
    if exists and not force:
        print(f"⏭️  Release {release_num} already exists in output directory")
        print(f"   Found {subject_count} subjects in merged release")
        print("   Skipping to avoid duplication")
        print("   (Use --force to re-merge anyway)")
        return subject_count, 0
    
    # Create output directory structure
    merged_release_dir = output_dir / f"release_{release_num}"
    dataset_folder = get_dataset_folder_name(release_num)
    merged_dataset_dir = merged_release_dir / dataset_folder
    
    if not dry_run:
        merged_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Track what we're copying
    subjects_merged = set()
    files_copied = 0
    conflicts = []
    
    # Process each split in order (primary first, then 2, 3, etc.)
    for split_name, split_path, split_num in split_paths:
        print(f"\n📁 Processing: {split_name} (split {split_num})")
        
        # Find dataset folder in this split
        split_dataset_dir = split_path / dataset_folder
        
        if not split_dataset_dir.exists():
            print(f"   ⚠️  Dataset folder not found: {split_dataset_dir}")
            continue
        
        # Copy metadata files (from the most complete split, usually the last one)
        if split_num >= 2:  # Prefer metadata from later splits
            metadata_files = [
                "dataset_description.json",
                "participants.tsv",
            ]
            # Also copy task-level JSON files
            for task_file in split_dataset_dir.glob("task-*.json"):
                metadata_files.append(task_file.name)
            
            for meta_file in metadata_files:
                src = split_dataset_dir / meta_file
                if src.exists():
                    dst = merged_dataset_dir / meta_file
                    if not dry_run:
                        shutil.copy2(src, dst)
                        print(f"   ✅ Copied metadata: {meta_file}")
                    else:
                        print(f"   📋 Would copy metadata: {meta_file}")
        
        # Copy all subject folders
        for subject_dir in split_dataset_dir.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
                continue
            
            subject_id = subject_dir.name
            merged_subject_dir = merged_dataset_dir / subject_id
            
            if subject_id in subjects_merged:
                print(f"   ⚠️  Subject {subject_id} already exists, merging files...")
                conflicts.append((subject_id, split_name))
                
                # Merge files from this subject
                if not dry_run:
                    merge_subject_files(subject_dir, merged_subject_dir)
            else:
                # First time seeing this subject, copy entire folder
                if not dry_run:
                    shutil.copytree(subject_dir, merged_subject_dir, dirs_exist_ok=True)
                    print(f"   ✅ Copied subject: {subject_id}")
                else:
                    print(f"   📋 Would copy subject: {subject_id}")
                
                subjects_merged.add(subject_id)
            
            # Count files
            if not dry_run:
                file_count = sum(1 for _ in subject_dir.rglob("*") if _.is_file())
                files_copied += file_count
    
    print(f"\n📊 Summary for Release {release_num}:")
    print(f"   Subjects merged: {len(subjects_merged)}")
    print(f"   Files copied: {files_copied}")
    if conflicts:
        print(f"   Conflicts resolved: {len(conflicts)}")
        for subject_id, split_name in conflicts[:5]:  # Show first 5
            print(f"      - {subject_id} (from {split_name})")
        if len(conflicts) > 5:
            print(f"      ... and {len(conflicts) - 5} more")
    
    return len(subjects_merged), files_copied

def main(data_dir="data", output_dir="data_merged", dry_run=False, force=False):
    """
    Main function to merge all split releases.
    
    Args:
        data_dir: Input data directory
        output_dir: Output directory for merged data
        dry_run: If True, only print what would be done
        force: If True, re-merge releases even if they already exist
    """
    print("="*70)
    print("EEG Challenge Data Merger")
    print("="*70)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be copied")
    if force:
        print("🔄 FORCE MODE - Will re-merge existing releases")
    print("="*70)
    
    # Find all split releases
    split_releases = find_split_releases(data_dir)
    
    if not split_releases:
        print("❌ No split releases found!")
        return
    
    print(f"\n📦 Found {len(split_releases)} release(s) with splits:")
    for release_num in sorted(split_releases.keys()):
        splits = split_releases[release_num]
        print(f"   Release {release_num}: {len(splits)} split(s)")
        for name, _, split_num in splits:
            print(f"      - {name} (split {split_num})")
    
    # Create output directory
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(exist_ok=True)
    
    # Merge each release
    total_subjects = 0
    total_files = 0
    skipped_releases = []
    
    for release_num in sorted(split_releases.keys()):
        split_paths = split_releases[release_num]
        subjects, files = merge_release(split_paths, output_path, release_num, dry_run=dry_run, force=force)
        if subjects > 0 and files == 0:
            # This means it was skipped (already exists)
            skipped_releases.append(release_num)
        total_subjects += subjects
        total_files += files
    
    print(f"\n{'='*70}")
    print("✅ Merge Complete!")
    print(f"{'='*70}")
    print(f"Total subjects merged: {total_subjects}")
    print(f"Total files processed: {total_files}")
    if skipped_releases:
        print(f"Skipped releases (already exist): {', '.join(f'R{r}' for r in skipped_releases)}")
    print(f"\nMerged data location: {output_path.resolve()}")
    
    if dry_run:
        print("\n💡 Run without --dry-run to actually merge the files")
    if skipped_releases and not force:
        print("\n💡 Use --force to re-merge skipped releases")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge split EEG challenge data releases"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_merged",
        help="Output directory for merged data (default: data_merged)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - only print what would be done"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-merge even if release already exists in output directory"
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        force=args.force
    )

