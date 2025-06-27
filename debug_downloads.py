#!/usr/bin/env python3
"""
Debug script to find downloaded batch result files and understand the disconnect.
"""

import os
import sys
import glob
import yaml

def main():
    # Load config to get checkpoint directory
    config_path = "config.yml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        print(f"📁 Config checkpoint_dir: {checkpoint_dir}")
    else:
        checkpoint_dir = "data/checkpoints"
        print(f"📁 Default checkpoint_dir: {checkpoint_dir}")
    
    print(f"\n🔍 Searching for batch result files...")
    
    # 1. Check the configured checkpoint directory
    print(f"\n1️⃣ Checking configured checkpoint directory: {checkpoint_dir}")
    if os.path.exists(checkpoint_dir):
        jsonl_files = glob.glob(os.path.join(checkpoint_dir, "*.jsonl"))
        if jsonl_files:
            print(f"   ✅ Found {len(jsonl_files)} JSONL files:")
            for file in jsonl_files:
                size = os.path.getsize(file)
                print(f"     - {os.path.basename(file)} ({size:,} bytes)")
        else:
            print(f"   ❌ No JSONL files found")
            
        # List all files in checkpoint directory
        all_files = os.listdir(checkpoint_dir)
        print(f"   📋 All files in {checkpoint_dir}:")
        for file in sorted(all_files):
            if os.path.isfile(os.path.join(checkpoint_dir, file)):
                print(f"     - {file}")
    else:
        print(f"   ❌ Directory does not exist")
    
    # 2. Search entire project for batch result files
    print(f"\n2️⃣ Searching entire project for batch result files...")
    project_root = "."
    found_files = []
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if "batch_result" in file.lower() or (file.endswith(".jsonl") and "batch" in file.lower()):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                found_files.append((full_path, size))
    
    if found_files:
        print(f"   ✅ Found {len(found_files)} potential batch result files:")
        for file_path, size in found_files:
            print(f"     - {file_path} ({size:,} bytes)")
    else:
        print(f"   ❌ No batch result files found in project")
    
    # 3. Search for any JSONL files that might be results
    print(f"\n3️⃣ Searching for all JSONL files...")
    jsonl_files = []
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".jsonl"):
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                jsonl_files.append((full_path, size))
    
    if jsonl_files:
        print(f"   ✅ Found {len(jsonl_files)} JSONL files:")
        for file_path, size in jsonl_files:
            print(f"     - {file_path} ({size:,} bytes)")
    else:
        print(f"   ❌ No JSONL files found in project")
    
    # 4. Check if there are any large files that might be results
    print(f"\n4️⃣ Searching for large files that might be batch results...")
    large_files = []
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                size = os.path.getsize(full_path)
                if size > 1000000:  # Files larger than 1MB
                    large_files.append((full_path, size))
            except:
                pass
    
    if large_files:
        # Sort by size, largest first
        large_files.sort(key=lambda x: x[1], reverse=True)
        print(f"   ✅ Found {len(large_files)} large files (>1MB):")
        for file_path, size in large_files[:10]:  # Show top 10
            print(f"     - {file_path} ({size:,} bytes)")
        if len(large_files) > 10:
            print(f"     ... and {len(large_files) - 10} more")
    else:
        print(f"   ❌ No large files found")

if __name__ == "__main__":
    main()