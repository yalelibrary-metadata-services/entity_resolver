#!/usr/bin/env python3
"""
Debug script to verify preprocessing worker configuration and file count.

This script simulates what happens when you run preprocessing to help diagnose
why only 5 workers are running instead of 32.
"""

import os
import logging
from src.config_utils import load_config_with_environment, get_environment

def debug_preprocessing_config():
    """Debug preprocessing worker configuration"""
    
    print("="*60)
    print("PREPROCESSING WORKER CONFIGURATION DEBUG")
    print("="*60)
    
    # Check environment
    env = get_environment()
    print(f"Environment detected: {env}")
    print(f"PIPELINE_ENV variable: {os.environ.get('PIPELINE_ENV', 'not set')}")
    
    # Load config
    config = load_config_with_environment('config.yml')
    
    # Extract key settings
    preprocessing_workers = config.get('preprocessing_workers')
    preprocessing_batch_size = config.get('preprocessing_batch_size')
    input_dir = config.get('input_dir', 'data/input')
    
    print(f"\nConfiguration values:")
    print(f"  preprocessing_workers: {preprocessing_workers}")
    print(f"  preprocessing_batch_size: {preprocessing_batch_size}")
    print(f"  input_dir: {input_dir}")
    
    # Check if input directory exists
    if os.path.exists(input_dir):
        # Count CSV files
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        print(f"\nInput directory analysis:")
        print(f"  Directory exists: {input_dir}")
        print(f"  Total CSV files: {len(csv_files)}")
        
        if len(csv_files) <= 10:
            print(f"  Files found: {csv_files}")
        else:
            print(f"  Sample files: {csv_files[:5]} ... (and {len(csv_files)-5} more)")
        
        # Calculate expected worker utilization
        if csv_files:
            optimal_batch_size = max(
                preprocessing_batch_size,  # Respect configured minimum
                preprocessing_workers * 3,  # At least 3 files per worker
                len(csv_files) // max(1, len(csv_files) // (preprocessing_workers * 4))
            )
            batch_size = min(optimal_batch_size, len(csv_files))
            
            print(f"\nBatch calculation:")
            print(f"  Base batch size: {preprocessing_batch_size}")
            print(f"  Optimal batch size: {optimal_batch_size}")
            print(f"  Actual batch size: {batch_size}")
            
            # Calculate workers per batch
            workers_per_batch = min(preprocessing_workers, batch_size)
            print(f"  Workers per batch: {workers_per_batch}/{preprocessing_workers}")
            print(f"  Worker utilization: {workers_per_batch/preprocessing_workers:.1%}")
            
            if workers_per_batch < preprocessing_workers:
                print(f"\n⚠️  EXPLANATION: Only {workers_per_batch} workers will be used because")
                print(f"   you only have {len(csv_files)} CSV files in {input_dir}")
                print(f"   ProcessPoolExecutor creates min(max_workers, tasks) processes")
                print(f"   To see all 32 workers, you need at least 32 CSV files")
        else:
            print(f"\n❌ No CSV files found in {input_dir}")
    else:
        print(f"\n❌ Input directory does not exist: {input_dir}")
    
    print(f"\n" + "="*60)

def show_expected_behavior():
    """Show what user should expect based on file count"""
    
    print("EXPECTED BEHAVIOR:")
    print("-" * 30)
    
    scenarios = [
        (5, "Current situation - only 5 workers active"),
        (16, "Half utilization - 16 workers active"), 
        (32, "Full utilization - all 32 workers active"),
        (64, "Still 32 workers (limited by max_workers)"),
    ]
    
    for file_count, description in scenarios:
        workers_used = min(32, file_count)  # min(max_workers, file_count)
        utilization = workers_used / 32
        print(f"  {file_count:2d} files → {workers_used:2d} workers ({utilization:.1%}) - {description}")
    
    print(f"\nTo fully utilize all 32 production workers:")
    print(f"  • Need at least 32 CSV files in your input directory")
    print(f"  • Each worker processes one file at a time") 
    print(f"  • With fewer files, excess workers remain idle")

if __name__ == "__main__":
    debug_preprocessing_config()
    print()
    show_expected_behavior()