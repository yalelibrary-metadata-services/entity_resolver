#!/usr/bin/env python3
"""
Weaviate Object Count Tool

Counts all existing objects in the EntityString collection.

Usage: 
  python weaviate-count-objects.py
"""

import weaviate
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateObjectCounter:
    """Simple tool to count objects in Weaviate EntityString collection"""
    
    def __init__(self):
        """Initialize the counter tool"""
        print("=== WEAVIATE OBJECT COUNTER ===")
        self.connect_weaviate()
    
    def connect_weaviate(self):
        """Connect to Weaviate with optimized timeouts for large datasets"""
        print("\n1. Connecting to Weaviate...")
        
        try:
            # Use custom connection with longer timeouts for large datasets
            from weaviate.connect import ConnectionParams
            
            # Configure for production with 13.8M objects
            connection_params = ConnectionParams.from_params(
                http_host="localhost",
                http_port=8080,
                http_secure=False,
                grpc_host="localhost",
                grpc_port=50051,
                grpc_secure=False
            )
            
            # Create client with massive scale timeouts for 45M object aggregations
            self.weaviate_client = weaviate.WeaviateClient(
                connection_params=connection_params,
                additional_config=weaviate.config.AdditionalConfig(
                    timeout=weaviate.config.Timeout(query=1800, insert=120)  # 30min for queries, 2min for inserts
                )
            )
            
            self.weaviate_client.connect()
            collections = self.weaviate_client.collections.list_all()
            print(f"   ✓ Connected to Weaviate with massive scale timeouts (30min query, 2min insert)")
            print(f"   ✓ Found {len(collections)} collections")
        except Exception as e:
            print(f"   ✗ Failed to connect to Weaviate: {e}")
            raise
    
    def count_objects(self):
        """Count all objects in the EntityString collection"""
        print("\n2. Counting objects in EntityString collection...")
        
        try:
            collection = self.weaviate_client.collections.get("EntityString")
            
            # Get total count of all objects
            total_count = collection.aggregate.over_all(
                total_count=True
            ).total_count
            
            print(f"   ✓ Total objects in EntityString collection: {total_count:,}")
            
            # Also count by field type for additional insight
            from weaviate.classes.query import Filter
            
            field_types = ["composite", "title", "person", "subjects", "genres", "attribution", "provision"]
            
            print("\n   Breakdown by field type (this may take several minutes for large datasets):")
            for i, field_type in enumerate(field_types):
                try:
                    print(f"     Counting {field_type}... ({i+1}/{len(field_types)})", end="", flush=True)
                    start_time = time.time()
                    
                    field_filter = Filter.by_property("field_type").equal(field_type)
                    field_count = collection.aggregate.over_all(
                        filters=field_filter,
                        total_count=True
                    ).total_count
                    
                    elapsed = time.time() - start_time
                    print(f" {field_count:,} (took {elapsed:.1f}s)")
                except Exception as e:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    print(f" Error after {elapsed:.1f}s - {e}")
            
            return total_count
            
        except Exception as e:
            print(f"   ✗ Error counting objects: {e}")
            return 0
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'weaviate_client'):
            try:
                self.weaviate_client.close()
            except:
                pass

def main():
    """Main function"""
    counter = WeaviateObjectCounter()
    try:
        total_count = counter.count_objects()
        print(f"\n=== SUMMARY ===")
        print(f"Total objects indexed: {total_count:,}")
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during operation: {e}")
        logger.exception("Operation failed")
    finally:
        counter.cleanup()

if __name__ == "__main__":
    main()