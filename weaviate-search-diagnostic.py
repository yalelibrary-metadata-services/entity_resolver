#!/usr/bin/env python3
"""
Weaviate Vector Search Diagnostic Tool

Performs near_vector searches using the composite field embedding of a given personId
or samples random title vectors and performs near_vector searches to explore dataset.
Shows subject information for matching results.

Usage: 
  python weaviate-search-diagnostic.py <personId> [--limit N] [--distance THRESHOLD]
  python weaviate-search-diagnostic.py --sample [--size N] [--search-limit N]
"""

import sys
import os
import pickle
import argparse
import logging
import random
from typing import Dict, List, Optional, Any
import weaviate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HASH_LOOKUP_PATHS = [
    'data/checkpoints/hash_lookup.pkl',
    'data/processed/hash_lookup.pkl'
]

DEFAULT_STRING_DICT_PATHS = [
    'data/checkpoints/string_dict.pkl',
    'data/processed/string_dict.pkl'
]

class WeaviateSearchDiagnostic:
    """Diagnostic tool for Weaviate vector searches"""
    
    def __init__(self, load_data=True):
        """Initialize the diagnostic tool"""
        print("=== WEAVIATE SEARCH DIAGNOSTIC ===")
        self.hash_lookup = None
        self.string_dict = None
        
        if load_data:
            self.load_data_structures()
        self.connect_weaviate()
    
    def find_file(self, search_paths: List[str]) -> Optional[str]:
        """Find the first existing file from a list of search paths"""
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None
    
    def load_data_structures(self):
        """Load hash lookup and string dictionary"""
        print("\n1. Loading data structures...")
        
        # Load hash lookup
        hash_lookup_path = self.find_file(DEFAULT_HASH_LOOKUP_PATHS)
        if not hash_lookup_path:
            raise FileNotFoundError(f"Could not find hash lookup. Searched: {DEFAULT_HASH_LOOKUP_PATHS}")
        
        with open(hash_lookup_path, 'rb') as f:
            self.hash_lookup = pickle.load(f)
        print(f"   ✓ Loaded hash lookup ({len(self.hash_lookup)} entries)")
        
        # Load string dictionary
        string_dict_path = self.find_file(DEFAULT_STRING_DICT_PATHS)
        if not string_dict_path:
            raise FileNotFoundError(f"Could not find string dictionary. Searched: {DEFAULT_STRING_DICT_PATHS}")
        
        with open(string_dict_path, 'rb') as f:
            self.string_dict = pickle.load(f)
        print(f"   ✓ Loaded string dictionary ({len(self.string_dict)} entries)")
    
    def connect_weaviate(self):
        """Connect to Weaviate"""
        print("\n2. Connecting to Weaviate...")
        
        try:
            self.weaviate_client = weaviate.connect_to_local()
            collections = self.weaviate_client.collections.list_all()
            print(f"   ✓ Connected to Weaviate")
            print(f"   ✓ Found {len(collections)} collections")
        except Exception as e:
            print(f"   ✗ Failed to connect to Weaviate: {e}")
            raise
    
    def get_composite_vector(self, person_id: str) -> Optional[List[float]]:
        """Get the composite field vector for a personId"""
        print(f"\n3. Retrieving composite vector for {person_id}...")
        
        # Get hash for this personId
        if person_id not in self.hash_lookup:
            print(f"   ✗ PersonId {person_id} not found in hash lookup")
            return None
        
        hashes = self.hash_lookup[person_id]
        if 'composite' not in hashes:
            print(f"   ✗ No composite hash found for {person_id}")
            return None
        
        composite_hash = hashes['composite']
        print(f"   Composite hash: {composite_hash}")
        
        # Get composite text preview
        composite_text = self.string_dict.get(composite_hash, "")
        print(f"   Composite text: {composite_text[:100]}...")
        
        # Query Weaviate for the vector
        try:
            collection = self.weaviate_client.collections.get("EntityString")
            
            from weaviate.classes.query import Filter
            
            hash_filter = Filter.by_property("hash_value").equal(composite_hash)
            field_filter = Filter.by_property("field_type").equal("composite")
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            query_result = collection.query.fetch_objects(
                filters=combined_filter,
                include_vector=True
            )
            
            if query_result.objects and len(query_result.objects) > 0:
                obj = query_result.objects[0]
                if hasattr(obj, 'vector'):
                    # Handle both dict and list vector formats
                    if isinstance(obj.vector, dict) and 'default' in obj.vector:
                        vector_data = obj.vector['default']
                    elif isinstance(obj.vector, list):
                        vector_data = obj.vector
                    else:
                        print(f"   ✗ Unexpected vector format: {type(obj.vector)}")
                        return None
                    
                    print(f"   ✓ Retrieved composite vector (dim={len(vector_data)})")
                    return vector_data
                else:
                    print(f"   ✗ No vector data in retrieved object")
                    return None
            else:
                print(f"   ✗ No objects found for composite hash {composite_hash}")
                return None
                
        except Exception as e:
            print(f"   ✗ Error retrieving composite vector: {e}")
            return None
    
    def perform_near_vector_search(self, vector: List[float], limit: int = 10, distance_threshold: float = None) -> List[Dict[str, Any]]:
        """Perform near_vector search using the provided vector"""
        print(f"\n4. Performing near_vector search (limit={limit})...")
        
        try:
            collection = self.weaviate_client.collections.get("EntityString")
            
            # Build query - always include distance for diagnostic purposes
            query_result = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                include_vector=False,
                return_metadata=['distance']
            )
            
            results = []
            
            for obj in query_result.objects:
                # Get the properties
                props = obj.properties
                hash_value = props.get('hash_value', '')
                field_type = props.get('field_type', '')
                
                # Only include composite field results
                if field_type == 'composite':
                    # Get the actual text content
                    if self.string_dict:
                        text_content = self.string_dict.get(hash_value, "")
                    else:
                        text_content = props.get('text_content', '')
                    
                    # Find the corresponding personId
                    person_id = None
                    if self.hash_lookup:
                        for pid, hashes in self.hash_lookup.items():
                            if hashes.get('composite') == hash_value:
                                person_id = pid
                                break
                    else:
                        person_id = props.get('person_id', 'Unknown')
                    
                    # Get distance from metadata if available
                    distance = 'N/A'
                    if hasattr(obj, 'metadata') and obj.metadata and hasattr(obj.metadata, 'distance'):
                        distance = obj.metadata.distance
                    
                    result = {
                        'person_id': person_id or 'Unknown',
                        'hash_value': hash_value,
                        'text_content': text_content,
                        'distance': distance
                    }
                    
                    # Apply distance threshold if specified
                    if distance_threshold is not None:
                        try:
                            dist = float(result['distance'])
                            if dist <= distance_threshold:
                                results.append(result)
                        except (ValueError, TypeError):
                            results.append(result)  # Include if distance is unknown
                    else:
                        results.append(result)
            
            print(f"   ✓ Found {len(results)} composite field matches")
            
            # Log distance for each result
            for i, result in enumerate(results, 1):
                print(f"      Result {i}: Distance={result['distance']}, PersonId={result['person_id']}")
            
            return results
            
        except Exception as e:
            print(f"   ✗ Error performing near_vector search: {e}")
            return []
    
    def get_random_title_vectors_and_search(self, sample_size: int = 10, search_limit: int = 5) -> List[Dict[str, Any]]:
        """Get random title vectors and perform near_vector searches with each"""
        print(f"\n2. Getting random sample of {sample_size} title vectors and performing searches...")
        
        try:
            collection = self.weaviate_client.collections.get("EntityString")
            
            # First, get title vectors
            from weaviate.classes.query import Filter
            title_filter = Filter.by_property("field_type").equal("title")
            
            # Get total count of title objects
            total_count = collection.aggregate.over_all(
                filters=title_filter,
                total_count=True
            ).total_count
            
            print(f"   Total title objects in index: {total_count:,}")
            
            if total_count == 0:
                print("   ✗ No title objects found in index")
                return []
            
            # Fetch title vectors for sampling
            fetch_limit = min(max(sample_size * 10, 1000), total_count)
            
            # Get title objects with their vectors
            query_result = collection.query.fetch_objects(
                filters=title_filter,
                limit=fetch_limit,
                include_vector=True
            )
            
            print(f"   Fetched {len(query_result.objects)} title objects for sampling")
            
            # Extract vectors and sample
            title_candidates = []
            for obj in query_result.objects:
                if hasattr(obj, 'vector'):
                    # Handle both dict and list vector formats
                    if isinstance(obj.vector, dict) and 'default' in obj.vector:
                        vector_data = obj.vector['default']
                    elif isinstance(obj.vector, list):
                        vector_data = obj.vector
                    else:
                        continue
                    
                    props = obj.properties
                    title_candidates.append({
                        'vector': vector_data,
                        'hash_value': props.get('hash_value', ''),
                        'text_content': props.get('text_content', '')
                    })
            
            # Randomly sample title vectors
            sample_size = min(sample_size, len(title_candidates))
            sampled_titles = random.sample(title_candidates, sample_size)
            
            print(f"   ✓ Selected {len(sampled_titles)} random title vectors")
            
            # Perform near_vector search for each sampled title
            all_results = []
            for i, title_obj in enumerate(sampled_titles, 1):
                title_text = title_obj['text_content']
                print(f"\n   🔍 Search {i}/{len(sampled_titles)}")
                print(f"   📖 Title: '{title_text}'")
                
                # Perform near_vector search
                search_results = collection.query.near_vector(
                    near_vector=title_obj['vector'],
                    limit=search_limit,
                    include_vector=False,
                    return_metadata=['distance']
                )
                
                # Process results and extract subject information
                search_matches = []
                for obj in search_results.objects:
                    props = obj.properties
                    hash_value = props.get('hash_value', '')
                    field_type = props.get('field_type', '')
                    text_content = props.get('text_content', '')
                    
                    # Get distance
                    distance = 'N/A'
                    if hasattr(obj, 'metadata') and obj.metadata and hasattr(obj.metadata, 'distance'):
                        distance = obj.metadata.distance
                    
                    # Find person_id and subjects if we have hash lookup
                    person_id = 'N/A'
                    subjects = []
                    
                    if self.hash_lookup:
                        for pid, hashes in self.hash_lookup.items():
                            if hashes.get(field_type) == hash_value:
                                person_id = pid
                                # Get subjects for this person
                                if 'subjects' in hashes and self.string_dict:
                                    subject_text = self.string_dict.get(hashes['subjects'], '')
                                    if subject_text.strip():
                                        subjects = [s.strip() for s in subject_text.split(';') if s.strip()]
                                break
                    
                    search_matches.append({
                        'person_id': person_id,
                        'field_type': field_type,
                        'hash_value': hash_value,
                        'text_content': text_content,
                        'distance': distance,
                        'subjects': subjects
                    })
                
                result_entry = {
                    'query_title': title_obj['text_content'],
                    'query_hash': title_obj['hash_value'],
                    'matches': search_matches
                }
                all_results.append(result_entry)
                
                print(f"     Found {len(search_matches)} matches")
            
            return all_results
            
        except Exception as e:
            print(f"   ✗ Error performing title vector sampling and search: {e}")
            return []

    def display_results(self, query_person_id: str, results: List[Dict[str, Any]]):
        """Display search results in a readable format"""
        print(f"\n=== SEARCH RESULTS for {query_person_id} ===")
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. PersonId: {result['person_id']}")
            print(f"   Distance: {result['distance']}")
            print(f"   Hash: {result['hash_value']}")
            print(f"   Content: {result['text_content'][:200]}...")
            if len(result['text_content']) > 200:
                print("   [Content truncated]")

    def display_sample_results(self, results: List[Dict[str, Any]]):
        """Display title vector search results in a readable format"""
        print(f"\n=== TITLE VECTOR SEARCH RESULTS ===")
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{'='*80}")
            print(f"🔍 SEARCH {i} - QUERY TITLE:")
            print(f"📖 '{result['query_title']}'")
            print(f"📋 Query Hash: {result['query_hash']}")
            print(f"🎯 Matches found: {len(result['matches'])}")
            print(f"{'='*80}")
            
            for j, match in enumerate(result['matches'], 1):
                print(f"\n   Match {j}:")
                print(f"     PersonId: {match['person_id']}")
                print(f"     Field Type: {match['field_type']}")
                print(f"     Distance: {match['distance']}")
                print(f"     Content: {match['text_content'][:150]}...")
                
                # Show subjects if available
                if match['subjects']:
                    print(f"     Subjects: {'; '.join(match['subjects'])}")
                else:
                    print(f"     Subjects: None")
                    
                if len(match['text_content']) > 150:
                    print("     [Content truncated]")
    
    def search(self, person_id: str, limit: int = 10, distance_threshold: float = None):
        """Main search function"""
        print(f"Query PersonId: {person_id}")
        print(f"Search Limit: {limit}")
        if distance_threshold:
            print(f"Distance Threshold: {distance_threshold}")
        
        # Ensure data structures are loaded for search functionality
        if self.hash_lookup is None or self.string_dict is None:
            self.load_data_structures()
        
        # Get the composite vector for the query personId
        query_vector = self.get_composite_vector(person_id)
        if not query_vector:
            print(f"\n✗ Could not retrieve composite vector for {person_id}")
            return
        
        # Perform the search
        results = self.perform_near_vector_search(query_vector, limit, distance_threshold)
        
        # Display results
        self.display_results(person_id, results)

    def sample(self, sample_size: int = 10, search_limit: int = 5):
        """Main sampling function - get random title vectors and search with them"""
        print(f"Random Title Sample Size: {sample_size}")
        print(f"Search Results per Title: {search_limit}")
        
        # Ensure data structures are loaded for subject extraction
        if self.hash_lookup is None or self.string_dict is None:
            print("Loading data structures for subject extraction...")
            self.load_data_structures()
        
        # Get random title vectors and perform searches
        results = self.get_random_title_vectors_and_search(sample_size, search_limit)
        
        # Display results
        self.display_sample_results(results)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'weaviate_client'):
            try:
                self.weaviate_client.close()
            except:
                pass

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Perform near_vector search using composite field embedding or get random samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vector search for a specific personId
  python weaviate-search-diagnostic.py 1605973#Agent700-17
  python weaviate-search-diagnostic.py 1605973#Agent700-17 --limit 20
  python weaviate-search-diagnostic.py 1605973#Agent700-17 --limit 50 --distance 0.3
  
  # Random title vector sampling with near_vector searches
  python weaviate-search-diagnostic.py --sample
  python weaviate-search-diagnostic.py --sample --size 5 --search-limit 10
        """
    )
    
    # Create mutually exclusive group for search vs sample
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('person_id', nargs='?', help='PersonId to search for')
    mode_group.add_argument('--sample', action='store_true', help='Get random sample of indexed objects')
    
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of search results (default: 10)')
    parser.add_argument('--distance', type=float, help='Maximum distance threshold for search results')
    parser.add_argument('--size', type=int, default=10, help='Sample size for random title sampling (default: 10)')
    parser.add_argument('--search-limit', type=int, default=5, help='Number of search results per title sample (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.sample and not args.person_id:
        parser.error('Either provide a person_id for search or use --sample for random sampling')
    
    # Create diagnostic tool - only load data structures for search mode
    load_data = not args.sample
    diagnostic = WeaviateSearchDiagnostic(load_data=load_data)
    try:
        if args.sample:
            # Perform random sampling
            diagnostic.sample(args.size, args.search_limit)
        else:
            # Perform vector search
            diagnostic.search(args.person_id, args.limit, args.distance)
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during operation: {e}")
        logger.exception("Operation failed")
    finally:
        diagnostic.cleanup()

if __name__ == "__main__":
    main()