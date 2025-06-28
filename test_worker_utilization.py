#!/usr/bin/env python3
"""
Test script to verify preprocessing worker utilization improvements.

This script simulates different file counts and shows how the new batch sizing
logic will utilize workers compared to the old approach.
"""

import os
import sys

def calculate_old_batch_utilization(total_files, base_batch_size, max_workers):
    """Calculate worker utilization with old fixed batch size approach"""
    batches = []
    for i in range(0, total_files, base_batch_size):
        batch_files = min(base_batch_size, total_files - i)
        workers_used = min(max_workers, batch_files)
        utilization = workers_used / max_workers
        batches.append({
            'batch_num': len(batches) + 1,
            'files': batch_files,
            'workers_used': workers_used,
            'utilization': utilization
        })
    return batches

def calculate_new_batch_utilization(total_files, base_batch_size, max_workers):
    """Calculate worker utilization with new dynamic batch size approach"""
    # New logic from preprocessing.py
    optimal_batch_size = max(
        base_batch_size,  # Respect configured minimum
        max_workers * 3,  # At least 3 files per worker for better utilization
        total_files // max(1, total_files // (max_workers * 4))  # Distribute evenly
    )
    
    batch_size = min(optimal_batch_size, total_files)
    
    batches = []
    for i in range(0, total_files, batch_size):
        batch_files = min(batch_size, total_files - i)
        workers_used = min(max_workers, batch_files)
        utilization = workers_used / max_workers
        batches.append({
            'batch_num': len(batches) + 1,
            'files': batch_files,
            'workers_used': workers_used,
            'utilization': utilization
        })
    
    return batches, batch_size, optimal_batch_size

def test_scenarios():
    """Test different file count scenarios"""
    
    # Production settings
    base_batch_size = 500  # prod preprocessing_batch_size
    max_workers = 32       # prod preprocessing_workers
    
    test_cases = [
        ("Small dataset", 5),
        ("Medium dataset", 50), 
        ("Large dataset", 500),
        ("Very large dataset", 2000)
    ]
    
    print("="*80)
    print("PREPROCESSING WORKER UTILIZATION ANALYSIS")
    print("="*80)
    print(f"Production Config: {max_workers} workers, {base_batch_size} base batch size")
    print("="*80)
    
    for name, total_files in test_cases:
        print(f"\n{name.upper()}: {total_files} files")
        print("-" * 60)
        
        # Old approach
        old_batches = calculate_old_batch_utilization(total_files, base_batch_size, max_workers)
        old_avg_utilization = sum(b['utilization'] for b in old_batches) / len(old_batches)
        old_total_workers = sum(b['workers_used'] for b in old_batches)
        
        # New approach  
        new_batches, new_batch_size, optimal_batch_size = calculate_new_batch_utilization(total_files, base_batch_size, max_workers)
        new_avg_utilization = sum(b['utilization'] for b in new_batches) / len(new_batches)
        new_total_workers = sum(b['workers_used'] for b in new_batches)
        
        print(f"OLD APPROACH (Fixed batch size: {base_batch_size}):")
        print(f"  Batches: {len(old_batches)}")
        print(f"  Average utilization: {old_avg_utilization:.1%}")
        print(f"  Total worker-tasks: {old_total_workers}")
        
        print(f"\nNEW APPROACH (Dynamic batch size: {new_batch_size}, optimal: {optimal_batch_size}):")
        print(f"  Batches: {len(new_batches)}")
        print(f"  Average utilization: {new_avg_utilization:.1%}")
        print(f"  Total worker-tasks: {new_total_workers}")
        
        improvement = (new_avg_utilization - old_avg_utilization) / old_avg_utilization * 100
        print(f"  IMPROVEMENT: {improvement:+.1f}% worker utilization")
        
        # Show batch details for small datasets
        if total_files <= 50:
            print(f"\n  Batch Details:")
            old_details = [f"{b['files']} files → {b['workers_used']} workers ({b['utilization']:.1%})" for b in old_batches]
            new_details = [f"{b['files']} files → {b['workers_used']} workers ({b['utilization']:.1%})" for b in new_batches]
            print(f"    OLD: {old_details}")
            print(f"    NEW: {new_details}")

if __name__ == "__main__":
    test_scenarios()