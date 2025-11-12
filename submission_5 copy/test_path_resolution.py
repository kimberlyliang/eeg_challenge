#!/usr/bin/env python3
"""Quick test to verify path resolution works"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from submission import resolve_path

print("Testing resolve_path...")
try:
    path = resolve_path('RandomForest_200_20251101_170036.pt')
    print(f"✅ Success! Found at: {path}")
    print(f"   File exists: {Path(path).exists()}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
