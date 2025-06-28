"""
Unified Embedding and Indexing Module for Entity Resolution

This module handles both the generation of vector embeddings and their direct indexing
in Weaviate, avoiding the storage of vectors on disk as per project requirements.
"""

import os
import sys
import logging
import pickle
import time
import json
import threading
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5

logger = logging.getLogger(__name__)

class EmbeddingAndIndexingPipeline:
    """
    Combined pipeline for embedding generation and direct Weaviate indexing.
    Implements a stateless vector processing approach that maintains checkpointing
    while avoiding intermediate vector storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary with embedding and indexing parameters
        """
        self.config = config
        
        # OpenAI API configuration for embeddings
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
            
        self.api_base = config.get("openai_api_base", "https://api.openai.com/v1")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dimensions = config.get("embedding_dimensions", 1536)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Rate limiting parameters
        self.batch_size = config.get("embedding_batch_size", 32)
        self.max_tokens_per_minute = config.get("max_tokens_per_minute", 5_000_000)
        self.max_requests_per_minute = config.get("max_requests_per_minute", 10_000)
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Embedding fields configuration
        self.embed_fields = config.get("embed_fields", ["composite", "person", "title"])
        self.skip_fields = config.get("skip_fields", ["provision", "subjects", "personId"])
        
        # Initialize Weaviate client
        self.weaviate_client = self._init_weaviate_client()
        
        # Prepare the collection
        self.collection = self._ensure_schema_exists()
        
        # Initialize checkpoint tracking
        self.processed_hashes = set()
        
        logger.info(f"Initialized EmbeddingAndIndexingPipeline with model {self.embedding_model}")
    
    def _init_weaviate_client(self):
        """Initialize and return a Weaviate client based on configuration."""
        # Get Weaviate connection parameters
        weaviate_url = self.config.get("weaviate_url", "http://localhost:8080")
        
        # Extract host and port information
        import urllib.parse
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
        grpc_port = self.config.get("weaviate_grpc_port", 50051)
        
        # Create API key authentication if provided
        auth_client_secret = None
        api_key = self.config.get("weaviate_api_key")
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
            
            logger.info(f"Connected to Weaviate at {weaviate_url} (gRPC port: {grpc_port})")
            return client
            
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            raise
    
    def _ensure_schema_exists(self):
        """Create Weaviate schema if it doesn't exist."""
        
        # Get recreate flag from config
        recreate_collections = self.config.get("recreate_collections", False)
        
        # Create collection with named vectors if it doesn't exist
        try:
            # If recreate_collections is true, we forcibly delete the collection
            if recreate_collections:
                try:
                    self.weaviate_client.collections.delete("EntityString")
                    logger.info("Deleted existing EntityString collection")
                    # Force collection to not exist by setting a flag
                    collection_exists = False
                except Exception as e:
                    logger.info(f"No existing EntityString collection found or cannot be deleted: {e}")
                    # We'll try to get it again below
                    collection_exists = True
            else:
                # Default case - we're not forcing recreation
                collection_exists = True
            
            # Check if collection exists only if we haven't just deleted it
            if collection_exists:
                try:
                    collection = self.weaviate_client.collections.get("EntityString")
                    logger.info("EntityString collection already exists")
                    return collection
                except Exception:
                    # Collection doesn't exist, we'll create it below
                    pass
            
            # At this point, we know we need to create the collection
            # Get index configuration parameters
            ef = self.config.get("weaviate_ef", 128)
            max_connections = self.config.get("weaviate_max_connections", 64)
            ef_construction = self.config.get("weaviate_ef_construction", 128)
            
            # Create collection with OpenAI Vectorizer configuration
            logger.info("Creating new EntityString collection with OpenAI Vectorizer")
            collection = self.weaviate_client.collections.create(
                name="EntityString",
                description="Collection for entity string values with their embeddings",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    dimensions=1536
                ),
                vector_index_config=Configure.VectorIndex.hnsw(
                    ef=ef,
                    max_connections=max_connections,
                    ef_construction=ef_construction,
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="original_string", data_type=DataType.TEXT),
                    Property(name="hash_value", data_type=DataType.TEXT),
                    Property(name="field_type", data_type=DataType.TEXT),
                    Property(name="frequency", data_type=DataType.INT)
                ]
            )
            
            logger.info(f"Created EntityString collection with OpenAI Vectorizer configuration")
            return collection
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {str(e)}")
            raise
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError))
    )
    def _get_embeddings_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get embeddings for a batch of texts from the OpenAI API.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts) where embeddings is a list of numpy arrays
            and token_counts is a list of token counts for each input
        """
        try:
            # Get embeddings using the client
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            # Extract embeddings and token count
            embeddings = []
            for embedding_data in response.data:
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            # Get token count
            token_count = response.usage.total_tokens
            token_counts = [token_count // len(texts)] * len(texts)  # Distribute evenly
            
            logger.debug(f"Generated {len(embeddings)} embeddings using {token_count} tokens")
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via OpenAI client: {str(e)}")
            # Re-raise to trigger retry
            raise requests.exceptions.RequestException(f"OpenAI API error: {str(e)}")

    
    def _index_embeddings_batch(self, items_to_index: List[Dict[str, Any]]) -> int:
        """
        Index a batch of embeddings directly in Weaviate with enhanced diagnostics.
        
        Args:
            items_to_index: List of items to index, each with hash_value, original_string,
                           field_type, frequency, and vector
        
        Returns:
            Number of successfully indexed items
        """
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.vector_diagnostics import VectorDiagnosticTool
        
        indexed_count = 0
        
        try:
            # Initialize diagnostic tool
            diagnostics = VectorDiagnosticTool(self.weaviate_client, {
                "embedding_dimensions": self.embedding_dimensions
            })
            
            # Select verification samples
            verification_samples = random.sample(
                items_to_index, 
                min(3, len(items_to_index))
            ) if items_to_index else []
            
            # Log batch details
            logger.info(f"Indexing batch of {len(items_to_index)} items with enhanced diagnostics")
            logger.info(f"Selected {len(verification_samples)} items for verification")
            
            # Use fixed-size batch configuration for better performance
            with self.collection.batch.fixed_size(batch_size=min(100, len(items_to_index))) as batch_writer:
                for item in items_to_index:
                    try:
                        # Generate UUID from hash value and field type for idempotency
                        uuid_input = f"{item['hash_value']}_{item['field_type']}"
                        uuid = generate_uuid5(uuid_input)
                        
                        # Remove vector from properties
                        properties = {k: v for k, v in item.items() if k != 'vector'}
                        
                        # Process vector with diagnostic tool
                        vector_data = diagnostics.debug_vector_transmission(
                            item['vector'],
                            item['hash_value'],
                            item['field_type']
                        )
                        
                        # Verify vector dimensions
                        if len(vector_data) != self.embedding_dimensions:
                            logger.error(f"Vector dimension mismatch: {len(vector_data)} != {self.embedding_dimensions}")
                            continue
                        
                        # Add object with explicit vector format
                        batch_writer.add_object(
                            properties=properties,
                            uuid=uuid,
                            vector=vector_data  # Direct vector assignment with normalized format
                        )
                        
                        indexed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error indexing item {item.get('hash_value', 'unknown')}: {str(e)}")
            
            # Verify vector persistence for sample items
            if verification_samples:
                verification_success = diagnostics.verify_vector_persistence(verification_samples)
                if verification_success:
                    logger.info("Vector persistence verification PASSED")
                else:
                    logger.warning("Vector persistence verification FAILED")
            
            logger.info(f"Successfully indexed {indexed_count}/{len(items_to_index)} items")
            
        except Exception as e:
            logger.error(f"Error in batch indexing with enhanced diagnostics: {str(e)}")
            # Don't raise here, allow processing to continue with other batches
        
        return indexed_count
    
    def _process_and_index_batch(self, batch_texts: List[Tuple[str, str, str, int]], 
                               lock: threading.Lock) -> Tuple[int, int]:
        """
        Process a batch of texts: generate embeddings and index in Weaviate.
        
        Args:
            batch_texts: List of (hash, text, field_type, frequency) tuples
            lock: Lock for thread-safe counter updates
            
        Returns:
            Tuple of (indexed_count, tokens_used)
        """
        # Rate limit enforcement
        current_time = time.time()
        with lock:
            if current_time - self.minute_start >= 60:
                # Reset counters for the new minute
                self.tokens_this_minute = 0
                self.requests_this_minute = 0
                self.minute_start = current_time
            elif (self.tokens_this_minute >= self.max_tokens_per_minute or 
                 self.requests_this_minute >= self.max_requests_per_minute):
                # Sleep until the next minute starts
                sleep_time = 60 - (current_time - self.minute_start)
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    self.tokens_this_minute = 0
                    self.requests_this_minute = 0
                    self.minute_start = time.time()
        
        # Extract hash values, texts, and field types
        hashes = [item[0] for item in batch_texts]
        texts = [item[1] for item in batch_texts]
        field_types = [item[2] for item in batch_texts]
        frequencies = [item[3] for item in batch_texts]
        
        try:
            # Get embeddings
            embeddings, token_counts = self._get_embeddings_batch(texts)
            
            # Update rate limiting counters
            with lock:
                self.requests_this_minute += 1
                self.tokens_this_minute += sum(token_counts)
            
            # Prepare items for indexing
            items_to_index = []
            for i, hash_val in enumerate(hashes):
                items_to_index.append({
                    'hash_value': hash_val,
                    'original_string': texts[i],
                    'field_type': field_types[i],
                    'frequency': frequencies[i],
                    'vector': embeddings[i]
                })
            
            # Index in Weaviate
            indexed_count = self._index_embeddings_batch(items_to_index)
            
            # Update processed hashes
            with lock:
                for hash_val in hashes:
                    self.processed_hashes.add(hash_val)
            
            return indexed_count, sum(token_counts)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return 0, 0
    
    def _process_batches_parallel(self, texts_to_process: List[Tuple[str, str, str, int]], 
                               max_workers: int = 4) -> Tuple[int, int]:
        """
        Process multiple batches in parallel using ThreadPoolExecutor.
        
        Args:
            texts_to_process: List of (hash, text, field_type, frequency) tuples
            max_workers: Maximum number of threads to use
            
        Returns:
            Tuple of (total_indexed, total_tokens)
        """
        # Calculate batch size for optimal parallelization
        num_texts = len(texts_to_process)
        
        # Use smaller batches for parallelization to optimize throughput
        adjusted_batch_size = min(self.batch_size, max(1, num_texts // max_workers))
        
        # Reset rate limits for this processing run
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Prepare batches
        batches = []
        for i in range(0, num_texts, adjusted_batch_size):
            batch = texts_to_process[i:i+adjusted_batch_size]
            batches.append(batch)
        
        logger.info(f"Processing {len(batches)} batches with {adjusted_batch_size} items per batch using {max_workers} workers")
        
        # Results container
        total_indexed = 0
        total_tokens = 0
        
        # Create a lock for synchronizing counter updates
        lock = threading.Lock()
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing
            futures = [executor.submit(self._process_and_index_batch, batch, lock) for batch in batches]
            
            # Process results as they complete with progress visualization
            with tqdm(total=len(futures), desc="Processing batch operations", unit="batch", ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for i, future in enumerate(as_completed(futures)):
                    indexed_count, tokens_used = future.result()
                    total_indexed += indexed_count
                    total_tokens += tokens_used
                    
                    # Update progress display with metrics
                    pbar.update(1)
                    pbar.set_postfix({
                        "indexed": total_indexed, 
                        "tokens": total_tokens,
                        "avg_tokens": int(total_tokens/max(1, total_indexed)) if total_indexed > 0 else 0
                    })
        
        return total_indexed, total_tokens
    
    def _select_strings_to_process(self, string_dict: Dict[str, str], 
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
        """
        Select strings for embedding and indexing based on field types.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            
        Returns:
            List of (hash, string, field_type, frequency) tuples to process
        """
        logger.info("Selecting strings to process based on field types")
        
        # Track field match statistics
        field_match_counts = {field: 0 for field in self.embed_fields}
        strings_to_process = []
        
        # Calculate cache coverage metrics
        total_candidates = sum(1 for hash_val in string_dict.keys() 
                            if hash_val in field_hash_mapping and 
                            any(field in self.embed_fields for field in field_hash_mapping[hash_val].keys()))
        
        already_processed = sum(1 for hash_val in string_dict.keys() 
                             if hash_val in self.processed_hashes and hash_val in field_hash_mapping and 
                             any(field in self.embed_fields for field in field_hash_mapping[hash_val].keys()))
        
        cache_coverage = 0 if total_candidates == 0 else already_processed / total_candidates
        
        # If we have almost complete coverage already, log and return empty list
        if total_candidates > 0 and cache_coverage >= 0.99:
            logger.info(f"===== PROCESSING STATUS: COMPLETE =====")
            logger.info(f"All eligible strings ({already_processed}/{total_candidates}) are already processed")
            logger.info(f"No new processing required - proceeding with existing data")
            return []
        
        # Process each string
        for hash_val, string_val in string_dict.items():
            # Skip already processed or invalid entries
            if hash_val in self.processed_hashes or hash_val == "NULL" or not string_val:
                continue
            
            # Get field mapping for this hash
            if hash_val in field_hash_mapping:
                # Extract fields
                fields = list(field_hash_mapping[hash_val].keys())
                
                # Check which fields match our embed_fields list
                matched_fields = [f for f in fields if f in self.embed_fields]
                
                # Update match counts
                for field in matched_fields:
                    field_match_counts[field] += 1
                
                # If any field matches, include this string - once per field type
                for field in matched_fields:
                    strings_to_process.append((
                        hash_val, 
                        string_val, 
                        field,
                        string_counts.get(hash_val, 1)
                    ))
        
        # Log selection statistics
        selected_count = len(strings_to_process)
        logger.info(f"Selected {selected_count} string-field pairs to process")
        logger.info(f"Processing status: {already_processed}/{total_candidates} strings already processed ({cache_coverage:.1%})")
        
        # Log field type distribution 
        for field, count in field_match_counts.items():
            logger.info(f"  Field '{field}': {count} strings selected")
        
        return strings_to_process
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load checkpoint data from disk.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
        """
        # Load processed hashes if checkpoint exists
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        if os.path.exists(processed_hashes_path):
            try:
                with open(processed_hashes_path, 'rb') as f:
                    self.processed_hashes = set(pickle.load(f))
                    
                logger.info(f"Loaded checkpoint with {len(self.processed_hashes)} processed hashes")
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.processed_hashes = set()
        else:
            logger.info(f"No checkpoint found at {processed_hashes_path}")
            self.processed_hashes = set()
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save checkpoint data to disk.
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save processed hashes
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        try:
            with open(processed_hashes_path, 'wb') as f:
                pickle.dump(list(self.processed_hashes), f)
                
            logger.info(f"Saved checkpoint with {len(self.processed_hashes)} processed hashes")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def process(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
              string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Process strings: generate embeddings and index in Weaviate.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with metrics about the process
        """
        logger.info("Starting unified embedding and indexing process")
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # Select strings to process
        strings_to_process = self._select_strings_to_process(
            string_dict, field_hash_mapping, string_counts
        )
        
        if not strings_to_process:
            logger.info("No new strings to process, proceeding with existing data")
            
            # Get collection stats
            try:
                result = self.collection.aggregate.over_all(total_count=True)
                collection_count = result.total_count
                logger.info(f"Collection contains {collection_count} objects")
                
            except Exception as e:
                logger.error(f"Error getting collection stats: {str(e)}")
                collection_count = None
                
            return {
                'status': 'completed',
                'elapsed_time': time.time() - start_time,
                'strings_processed': 0,
                'total_tokens': 0,
                'collection_count': collection_count
            }
        
        # Calculate optimal number of workers based on available CPUs
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        max_workers = min(available_cores, self.config.get("embedding_workers", 4))
        logger.info(f"Using {max_workers} worker threads on system with {available_cores} cores")
        
        # Process in batches with checkpoints
        checkpoint_batch = self.config.get("checkpoint_batch", 1000)
        total_processed = 0
        total_tokens = 0
        
        # Master progress bar for overall process
        with tqdm(total=len(strings_to_process), desc="Embedding and indexing", unit="item", ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            # Process in checkpoint batches
            for i in range(0, len(strings_to_process), checkpoint_batch):
                batch = strings_to_process[i:i+checkpoint_batch]
                batch_start_time = time.time()
                
                # Process batch
                indexed_count, tokens_used = self._process_batches_parallel(batch, max_workers)
                
                # Update counters
                total_processed += indexed_count
                total_tokens += tokens_used
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Calculate and display throughput metrics
                batch_time = time.time() - batch_start_time
                items_per_sec = indexed_count / batch_time if batch_time > 0 else 0
                
                # Update progress bar with metrics
                pbar.set_postfix({
                    "indexed": total_processed, 
                    "tokens": total_tokens,
                    "rate": f"{items_per_sec:.1f}/s"
                })
                
                # Save checkpoint
                self.save_checkpoint(checkpoint_dir)
                
                # Log detailed progress 
                progress_pct = total_processed / len(strings_to_process) * 100
                logger.info(f"\n{'='*30} CHECKPOINT SUMMARY {'='*30}")
                logger.info(f"Processed batch of {len(batch)} strings in {batch_time:.2f}s ({items_per_sec:.2f} items/sec)")
                logger.info(f"Progress: {total_processed}/{len(strings_to_process)} strings ({progress_pct:.2f}%)")
                if indexed_count > 0:
                    logger.info(f"Tokens used: {tokens_used} ({tokens_used/indexed_count:.1f} per item)")
                else:
                    logger.info(f"Tokens used: {tokens_used} (N/A per item)")
        
        elapsed_time = time.time() - start_time
        overall_throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"\n{'='*30} PROCESS COMPLETE {'='*30}")
        logger.info(f"Embedding and indexing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {total_processed} strings using {total_tokens} tokens")
        logger.info(f"Overall throughput: {overall_throughput:.2f} items/sec")
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"Collection now contains {collection_count} objects")
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        # Return metrics without closing the client here (it's closed in the parent function)
        return {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'strings_processed': total_processed,
            'tokens_used': total_tokens,
            'collection_count': collection_count,
            'throughput': overall_throughput
        }

def embedding_and_indexing(config: Dict[str, Any], string_dict: Dict[str, str], 
                         field_hash_mapping: Dict[str, Dict[str, int]],
                         string_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Unified function for embedding generation and indexing.
    
    Args:
        config: Configuration dictionary
        string_dict: Dictionary mapping hash to string value
        field_hash_mapping: Mapping of hash to field types
        string_counts: Mapping of hash to frequency count
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Starting unified embedding and indexing process")
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize the pipeline
    pipeline = None
    metrics = {}
    
    try:
        pipeline = EmbeddingAndIndexingPipeline(config)
        
        # Process the data
        metrics = pipeline.process(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
    except Exception as e:
        logger.error(f"Error in embedding and indexing pipeline: {str(e)}")
        metrics = {
            'status': 'error',
            'error': str(e),
            'elapsed_time': 0
        }
    finally:
        # Ensure client is closed even if an exception occurs
        if pipeline is not None and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
                logger.info("Weaviate client connection closed in cleanup handler")
            except Exception as e:
                logger.error(f"Error closing Weaviate client in cleanup: {str(e)}")
    
    return metrics

def main(config_path: str = 'config.yml'):
    """
    Main function for unified embedding and indexing.
    
    Args:
        config_path: Path to the configuration file
    """
    import yaml
    import pickle
    import os
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(config_path)
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Load preprocessing data
    string_dict_path = os.path.join(checkpoint_dir, "string_dict.pkl")
    field_hash_mapping_path = os.path.join(checkpoint_dir, "field_hash_mapping.pkl") 
    string_counts_path = os.path.join(checkpoint_dir, "string_counts.pkl")
    
    try:
        # Load string dictionary
        with open(string_dict_path, 'rb') as f:
            string_dict = pickle.load(f)
        
        # Load field hash mapping
        with open(field_hash_mapping_path, 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        # Load string counts
        with open(string_counts_path, 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings, "
                   f"{len(field_hash_mapping)} field mappings, {len(string_counts)} string counts")
        
        # Run embedding and indexing
        metrics = embedding_and_indexing(config, string_dict, field_hash_mapping, string_counts)
        
        # Save metrics
        output_dir = os.path.join(config.get("output_dir", "data/output"))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "embedding_indexing_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_dir}/embedding_indexing_metrics.json")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified embedding and indexing for entity resolution')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start from beginning')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handle reset flag
    if args.reset:
        import shutil
        import os
        
        # Load config to get checkpoint directory with environment-specific overrides
        from src.config_utils import load_config_with_environment
        config = load_config_with_environment(args.config)
        
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        # Delete processed hashes checkpoint
        if os.path.exists(processed_hashes_path):
            os.remove(processed_hashes_path)
            logger.info(f"Deleted checkpoint file: {processed_hashes_path}")
    
    # Run unified embedding and indexing
    main(args.config)
