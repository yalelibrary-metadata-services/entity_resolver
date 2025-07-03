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
            # Use the proper weaviate.connect_to_local() method with timeout overrides
            
            # Use connect_to_local with massive timeout configurations
            self.weaviate_client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
                # Configure timeouts for massive scale operations
                additional_config=weaviate.config.AdditionalConfig(
                    timeout=weaviate.config.Timeout(
                        query=1800,    # 30min for queries - THIS IS THE KEY FIX
                        insert=120,    # 2min for inserts
                        init=60,       # 1min for initialization
                        close=30       # 30sec for cleanup
                    ),
                    connection_pool_maxsize=100
                ),
                skip_init_checks=False
            )
            
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
            
            print("\n   Breakdown by field type (using sampling for massive datasets):")
            print("   Note: For 13.8M+ objects, using approximate counts via sampling...")
            
            for i, field_type in enumerate(field_types):
                try:
                    print(f"     Counting {field_type}... ({i+1}/{len(field_types)})", end="", flush=True)
                    start_time = time.time()
                    
                    # Alternative approach: Use GraphQL with limit for sampling
                    # This is much faster than full aggregation on massive datasets
                    field_filter = Filter.by_property("field_type").equal(field_type)
                    
                    # First try a limited query to see if any exist
                    sample_query = collection.query.fetch_objects(
                        filters=field_filter,
                        limit=1
                    )
                    
                    if len(sample_query.objects) == 0:
                        # No objects of this type
                        elapsed = time.time() - start_time
                        print(f" 0 (took {elapsed:.1f}s)")
                    else:
                        # Try the full count with extended timeout handling
                        try:
                            field_count = collection.aggregate.over_all(
                                filters=field_filter,
                                total_count=True
                            ).total_count
                            
                            elapsed = time.time() - start_time
                            print(f" {field_count:,} (took {elapsed:.1f}s)")
                        except Exception as count_error:
                            # If count fails, estimate based on sampling
                            elapsed = time.time() - start_time
                            print(f" >1 (count failed after {elapsed:.1f}s, sampling needed)")
                            
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