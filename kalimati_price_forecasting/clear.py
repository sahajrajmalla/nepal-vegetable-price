#!/usr/bin/env python3
"""
clear.py
========
Clears all auto-generated files (models, figures, reports, logs, cached data)
to ensure a completely fresh run of the pipeline.

This script ensures your `outputs/` folder and cache directories are completely 
emptied so `run_all.py` can generate new results from scratch.

It will NOT delete your raw data or configuration files.
"""

import shutil
from pathlib import Path

def clear_directory(dir_path: Path):
    """Deletes all contents of a directory but keeps the directory itself."""
    if not dir_path.exists():
        return
        
    deleted_items = 0
    for item in dir_path.iterdir():
        if item.name == ".gitkeep": # Preserve gitkeeps if they exist
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            deleted_items += 1
        except Exception as e:
            print(f"Failed to delete {item}: {e}")
            
    if deleted_items > 0:
        print(f"✓ Cleared {deleted_items} items from: {dir_path.resolve()}")

def main():
    root = Path(__file__).resolve().parent

    print("🧹 Starting cleanup of auto-generated pipeline files...\n")

    # Outputs
    dirs_to_clear = [
        root / "outputs" / "cleaned_data",
        root / "outputs" / "figures",
        root / "outputs" / "models",
        root / "outputs" / "reports",
        root / "lightning_logs",  # NeuralForecast/PyTorch Lightning logs
        root / "data" / "interim",
        root / "data" / "processed",
    ]

    # Also check if processed data lives one directory up (due to raw_dir config)
    dirs_to_clear.extend([
        (root / ".." / "data" / "processed").resolve(),
        (root / ".." / "data" / "interim").resolve()
    ])

    # De-duplicate directories (in case paths resolve to the same location)
    dirs_to_clear = list(set(dirs_to_clear))

    for d in dirs_to_clear:
        clear_directory(d)

    # ── Clean up Python caches ──
    pycache_count = 0
    for pycache in root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            pycache_count += 1
        except Exception:
            pass
            
    if pycache_count > 0:
        print(f"✓ Removed {pycache_count} __pycache__ directories.")

    print("\n✨ Cleanup complete! The workspace is ready for a fresh PROD run.")
    print("   Execute: python3 run_all.py --mode PROD")

if __name__ == "__main__":
    main()
