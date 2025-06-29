#!/usr/bin/env python3
"""
Simple diagnostic script to check the number of objects in Weaviate EntityString collection.
"""

import os
import sys
import yaml
import weaviate

def load_config(config_path: str = "config.yml"):
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def check_weaviate_count(config):
    """Check the number of objects in EntityString collection."""
    # Get Weaviate URL from config
    weaviate_url = config.get("weaviate_url", "http://localhost:8080")
    
    print(f"🔗 Connecting to Weaviate at: {weaviate_url}")
    
    # Simple connection using connect_to_local for localhost
    if "localhost" in weaviate_url or "127.0.0.1" in weaviate_url:
        client = weaviate.connect_to_local()
    else:
        client = weaviate.connect_to_custom(
            http_host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            http_port=8080,
            http_secure=False,
            grpc_port=50051,
            grpc_secure=False
        )
    
    try:
        # Check if client is ready
        if not client.is_ready():
            print("❌ Weaviate is not ready")
            return
            
        # Check if EntityString collection exists
        if not client.collections.get("EntityString"):
            print("ℹ️  EntityString collection does not exist")
            return
            
        # Get collection and count objects
        collection = client.collections.get("EntityString")
        response = collection.aggregate.over_all(total_count=True)
        
        if response and hasattr(response, 'total_count'):
            count = response.total_count
            print(f"✅ EntityString collection contains: {count:,} objects")
        else:
            print("⚠️  Could not retrieve count from collection")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Weaviate EntityString collection count')
    parser.add_argument('--config', default='config.yml', help='Config file path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    check_weaviate_count(config)

if __name__ == "__main__":
    main()