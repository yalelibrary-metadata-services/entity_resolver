#!/usr/bin/env python3
"""
Fix Jina v3 cache corruption issue and prepare environment
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def clear_jina_cache():
    """Clear corrupted Jina model cache."""
    cache_dirs = [
        "~/.cache/huggingface/modules/transformers_modules/jinaai",
        "~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3",
        "~/.cache/torch/sentence_transformers/jinaai_jina-embeddings-v3"
    ]
    
    for cache_dir in cache_dirs:
        expanded_path = os.path.expanduser(cache_dir)
        if os.path.exists(expanded_path):
            print(f"Removing corrupted cache: {expanded_path}")
            shutil.rmtree(expanded_path)
    
    print("Cache cleared successfully")


def install_dependencies():
    """Install required dependencies."""
    print("Installing einops...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "einops>=0.7.0"])
    print("einops installed successfully")


def test_jina_loading():
    """Test if Jina v3 can be loaded."""
    print("\nTesting Jina v3 loading...")
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading Jina v3 model (this may take a few minutes)...")
        model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
        print("✅ Jina v3 loaded successfully!")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✅ Test encoding successful! Embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load Jina v3: {e}")
        return False


def main():
    print("Fixing Jina v3 cache and environment issues...\n")
    
    # Step 1: Clear corrupted cache
    clear_jina_cache()
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Test loading
    success = test_jina_loading()
    
    if success:
        print("\n✅ Environment fixed! You can now run your training script.")
        print("\nExample command:")
        print("CUDA_VISIBLE_DEVICES=0 python train_setfit_simple.py \\")
        print("    --model_name 'jinaai/jina-embeddings-v3' \\")
        print("    --csv_path ../data/input/training_dataset.csv \\")
        print("    --ground_truth_path ../data/output/updated_identity_classification_map_v6_pruned.json \\")
        print("    --output_dir ./model_jina_v3 \\")
        print("    --batch_size 16 \\")
        print("    --epochs 2")
    else:
        print("\n❌ Still having issues. Try these alternatives:")
        print("1. Use a different model (e.g., sentence-transformers/all-mpnet-base-v2)")
        print("2. Update your packages: pip install -U transformers sentence-transformers setfit")
        print("3. Use the fallback script: train_setfit_fallback.py")


if __name__ == "__main__":
    main()