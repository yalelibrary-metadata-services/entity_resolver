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
    try:
        # Get Weaviate connection details from config
        weaviate_config = config.get('weaviate', {})
        url = weaviate_config.get('url', 'http://localhost:8080')
        
        print(f"🔗 Connecting to Weaviate at: {url}")
        
        # Connect to Weaviate
        client = weaviate.Client(url=url)
        
        # Check if collection exists
        try:
            # Get collection info
            collection = client.collections.get("EntityString")
            
            # Query for total count
            response = collection.aggregate.over_all(total_count=True)
            
            if response and hasattr(response, 'total_count'):
                count = response.total_count
                print(f"✅ EntityString collection contains: {count:,} objects")
            else:
                print("⚠️  Could not retrieve count from collection")
                
        except Exception as collection_error:
            if "not found" in str(collection_error).lower() or "does not exist" in str(collection_error).lower():
                print("ℹ️  EntityString collection does not exist")
            else:
                print(f"❌ Error accessing collection: {collection_error}")
        
        # Close connection
        client.close()
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        sys.exit(1)

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