#!/usr/bin/env python3
"""
Weaviate Object Count Tool

Counts all existing objects in the EntityString collection.

Usage: 
  python weaviate-count-objects.py
"""

import weaviate
import logging

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
        """Connect to Weaviate"""
        print("\n1. Connecting to Weaviate...")
        
        try:
            self.weaviate_client = weaviate.connect_to_local()
            collections = self.weaviate_client.collections.list_all()
            print(f"   ✓ Connected to Weaviate")
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
            
            print("\n   Breakdown by field type:")
            for field_type in field_types:
                try:
                    field_filter = Filter.by_property("field_type").equal(field_type)
                    field_count = collection.aggregate.over_all(
                        filters=field_filter,
                        total_count=True
                    ).total_count
                    print(f"     {field_type}: {field_count:,}")
                except Exception as e:
                    print(f"     {field_type}: Error - {e}")
            
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