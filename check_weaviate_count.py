#!/usr/bin/env python3
"""
Simple diagnostic script to check the number of objects in Weaviate EntityString collection.
"""

import os
import sys
import yaml
import weaviate
import urllib.parse

def load_config(config_path: str = "config.yml"):
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

def _init_weaviate_client(config):
    """Initialize Weaviate client following the same pattern as existing modules."""
    # Get Weaviate connection parameters
    weaviate_url = config.get("weaviate_url", "http://localhost:8080")
    
    # Extract host and port information
    parsed_url = urllib.parse.urlparse(weaviate_url)
    
    # Extract host (without protocol)
    host = parsed_url.netloc
    if ':' in host:
        host, port_str = host.split(':', 1)
        port = int(port_str)
    else:
        port = 8080  # Default HTTP port
    
    # Determine if secure connection (HTTPS)
    secure = parsed_url.scheme == 'https'
    
    # Default gRPC port is typically 50051
    grpc_port = config.get("weaviate_grpc_port", 50051)
    
    # Create API key authentication if provided
    auth_client_secret = None
    api_key = config.get("weaviate_api_key")
    if api_key:
        from weaviate.auth import AuthApiKey
        auth_client_secret = AuthApiKey(api_key)
    
    try:
        # Create connection parameters
        from weaviate.connect import ConnectionParams
        connection_params = ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,  # Using same host for gRPC
            grpc_port=grpc_port,
            grpc_secure=secure  # Using same security setting for gRPC
        )
        
        # Initialize client
        client = weaviate.WeaviateClient(
            connection_params=connection_params,
            auth_client_secret=auth_client_secret
        )
        
        # Connect to Weaviate
        client.connect()
        
        print(f"🔗 Connected to Weaviate at {weaviate_url}")
        return client
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        raise

def check_weaviate_count(config):
    """Check the number of objects in EntityString collection."""
    client = None
    try:
        # Initialize Weaviate client
        client = _init_weaviate_client(config)
        
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
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        # Close connection
        if client:
            try:
                client.close()
            except:
                pass

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