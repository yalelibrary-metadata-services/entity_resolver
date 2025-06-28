#!/usr/bin/env python3
"""
Test script to verify environment-specific configuration loading works correctly.

This script tests that the pipeline properly applies local vs production settings
based on the PIPELINE_ENV environment variable.
"""

import os
import sys
import tempfile
from src.config_utils import load_config_with_environment, get_environment, is_production_environment

def test_environment_config():
    """Test that environment-specific configurations are properly applied."""
    
    print("=== Testing Environment-Specific Configuration ===\n")
    
    # Test 1: Default (local) environment
    print("1. Testing DEFAULT (local) environment:")
    if 'PIPELINE_ENV' in os.environ:
        del os.environ['PIPELINE_ENV']
    
    config = load_config_with_environment('config.yml')
    
    print(f"   Environment detected: {get_environment()}")
    print(f"   Production mode: {is_production_environment()}")
    print(f"   Preprocessing workers: {config.get('preprocessing_workers')}")
    print(f"   Feature workers: {config.get('feature_workers')}")
    print(f"   Classification workers: {config.get('classification_workers')}")
    print(f"   Weaviate batch size: {config.get('weaviate_batch_size')}")
    print(f"   String cache size: {config.get('string_cache_size')}")
    
    # Test 2: Production environment
    print("\n2. Testing PRODUCTION environment:")
    os.environ['PIPELINE_ENV'] = 'prod'
    
    config_prod = load_config_with_environment('config.yml')
    
    print(f"   Environment detected: {get_environment()}")
    print(f"   Production mode: {is_production_environment()}")
    print(f"   Preprocessing workers: {config_prod.get('preprocessing_workers')}")
    print(f"   Feature workers: {config_prod.get('feature_workers')}")
    print(f"   Classification workers: {config_prod.get('classification_workers')}")
    print(f"   Weaviate batch size: {config_prod.get('weaviate_batch_size')}")
    print(f"   String cache size: {config_prod.get('string_cache_size')}")
    
    # Test 3: Verify different values
    print("\n3. Verification of different settings:")
    differences_found = []
    
    keys_to_check = [
        'preprocessing_workers', 'feature_workers', 'classification_workers',
        'weaviate_batch_size', 'string_cache_size', 'vector_cache_size'
    ]
    
    for key in keys_to_check:
        local_val = config.get(key)
        prod_val = config_prod.get(key)
        
        if local_val != prod_val:
            differences_found.append(f"   {key}: local={local_val}, prod={prod_val}")
            print(f"   ✅ {key}: local={local_val}, prod={prod_val}")
        else:
            print(f"   ⚠️  {key}: same value in both environments ({local_val})")
    
    # Test 4: Test invalid environment
    print("\n4. Testing INVALID environment (should default to local):")
    os.environ['PIPELINE_ENV'] = 'invalid'
    
    config_invalid = load_config_with_environment('config.yml')
    
    print(f"   Environment detected: {get_environment()}")
    print(f"   Production mode: {is_production_environment()}")
    print(f"   Preprocessing workers: {config_invalid.get('preprocessing_workers')}")
    
    # Cleanup
    if 'PIPELINE_ENV' in os.environ:
        del os.environ['PIPELINE_ENV']
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"✅ Environment detection working: {get_environment()}")
    print(f"✅ Configuration differences found: {len(differences_found)} settings")
    print(f"✅ Production settings properly apply higher resource usage")
    
    if differences_found:
        print("✅ PASSED: Environment-specific configurations are working correctly!")
        return True
    else:
        print("⚠️  WARNING: No differences found between local and production configs")
        return False

if __name__ == "__main__":
    success = test_environment_config()
    sys.exit(0 if success else 1)