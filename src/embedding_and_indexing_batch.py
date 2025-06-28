"""
Batch Embedding and Indexing Module for Entity Resolution

This module implements OpenAI's Batch API for embedding generation, providing
cost-effective processing at 50% lower cost with 24-hour turnaround time.
Separates embedding generation from indexing to handle the asynchronous nature
of batch processing.
"""

import os
import sys
import logging
import pickle
import time
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using fallback token estimation")

class BatchJobStatus:
    """Enum-like class for batch job statuses."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class BatchEmbeddingPipeline:
    """
    Batch-based pipeline for embedding generation using OpenAI's Batch API.
    Provides cost-effective processing with 50% lower cost and 24-hour turnaround.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the batch pipeline with configuration.
        
        Args:
            config: Configuration dictionary with batch processing parameters
        """
        self.config = config
        self.weaviate_client = None  # Initialize as None first
        self.collection = None
        
        # OpenAI API configuration
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
            
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dimensions = config.get("embedding_dimensions", 1536)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Batch processing parameters
        self.batch_size = config.get("batch_embedding_size", 50000)  # Larger batches for cost efficiency
        self.max_requests_per_file = config.get("max_requests_per_file", 50000)
        self.manual_polling = config.get("batch_manual_polling", True)  # Manual polling by default
        self.poll_interval = config.get("batch_poll_interval", 300)  # 5 minutes (only used in auto mode)
        self.max_wait_time = config.get("batch_max_wait_time", 86400)  # 24 hours
        
        # Token quota management
        self.token_quota_limit = config.get("token_quota_limit", 500_000_000)  # 500M default OpenAI limit
        self.quota_safety_margin = config.get("quota_safety_margin", 0.1)  # 10% safety margin
        self.max_concurrent_jobs = config.get("max_concurrent_jobs", 50)  # Limit concurrent jobs
        
        # Request quota management (1M enqueued requests limit)
        self.request_quota_limit = config.get("request_quota_limit", 1_000_000)  # 1M default OpenAI limit
        self.max_requests_per_job = config.get("max_requests_per_job", 50_000)  # Safety limit per job
        
        # Embedding fields configuration
        self.embed_fields = config.get("embed_fields", ["composite", "person", "title"])
        self.skip_fields = config.get("skip_fields", ["provision", "subjects", "personId"])
        
        # Initialize Weaviate client
        self.weaviate_client = self._init_weaviate_client()
        
        # Prepare the collection
        self.collection = self._ensure_schema_exists()
        
        # Initialize batch job tracking
        self.batch_jobs = {}  # job_id -> job_info
        self.processed_hashes = set()
        
        logger.info(f"Initialized BatchEmbeddingPipeline with model {self.embedding_model}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
    
    def close(self):
        """Properly close all connections."""
        if self.weaviate_client is not None:
            try:
                self.weaviate_client.close()
                logger.debug("Weaviate client connection closed")
            except Exception as e:
                logger.debug(f"Error closing Weaviate client: {e}")
            finally:
                self.weaviate_client = None
                self.collection = None
    
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
                    collection_exists = False
                except Exception as e:
                    logger.info(f"No existing EntityString collection found or cannot be deleted: {e}")
                    collection_exists = True
            else:
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
    
    def _create_batch_requests_file(self, strings_to_process: List[Tuple[str, str, str, int]], 
                                  output_path: str) -> Dict[str, Any]:
        """
        Create a JSONL file with batch embedding requests.
        
        Args:
            strings_to_process: List of (hash, text, field_type, frequency) tuples
            output_path: Path to save the JSONL file
            
        Returns:
            Dictionary with file metadata
        """
        logger.info(f"Creating batch requests file with {len(strings_to_process)} requests")
        
        # Create custom ID mapping for tracking
        custom_id_mapping = {}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (hash_val, text, field_type, frequency) in enumerate(strings_to_process):
                # Create unique custom ID
                custom_id = f"{hash_val}_{field_type}_{i}"
                custom_id_mapping[custom_id] = {
                    'hash_value': hash_val,
                    'original_string': text,
                    'field_type': field_type,
                    'frequency': frequency,
                    'index': i
                }
                
                # Create batch request
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.embedding_model,
                        "input": text,
                        "encoding_format": "float"
                    }
                }
                
                # Write as JSONL
                f.write(json.dumps(request) + '\n')
        
        file_metadata = {
            'path': output_path,
            'request_count': len(strings_to_process),
            'custom_id_mapping': custom_id_mapping,
            'created_at': time.time()
        }
        
        logger.info(f"Created batch requests file: {output_path} ({len(strings_to_process)} requests)")
        return file_metadata
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _upload_batch_file(self, file_path: str) -> str:
        """
        Upload batch requests file to OpenAI.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            File ID from OpenAI
        """
        logger.info(f"Uploading batch file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                file_response = self.openai_client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            logger.info(f"Successfully uploaded file, ID: {file_response.id}")
            return file_response.id
            
        except Exception as e:
            logger.error(f"Error uploading batch file: {str(e)}")
            raise
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _create_batch_job(self, input_file_id: str, description: str = None) -> str:
        """
        Create a batch job for processing embeddings.
        
        Args:
            input_file_id: File ID from OpenAI upload
            description: Optional job description
            
        Returns:
            Batch job ID
        """
        logger.info(f"Creating batch job for file: {input_file_id}")
        
        try:
            batch_response = self.openai_client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": description or "Entity resolution embeddings batch job",
                    "created_by": "embedding_and_indexing_batch"
                }
            )
            
            logger.info(f"Created batch job: {batch_response.id}")
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Error creating batch job: {str(e)}")
            raise
    
    def _poll_batch_job(self, batch_id: str) -> Dict[str, Any]:
        """
        Poll batch job status until completion.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Final batch job status
        """
        logger.info(f"Polling batch job: {batch_id}")
        
        start_time = time.time()
        last_status = None
        
        with tqdm(desc=f"Waiting for batch job {batch_id[:8]}...", 
                 bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
            
            while time.time() - start_time < self.max_wait_time:
                try:
                    batch_status = self.openai_client.batches.retrieve(batch_id)
                    current_status = batch_status.status
                    
                    # Update progress if status changed
                    if current_status != last_status:
                        logger.info(f"Batch job {batch_id} status: {current_status}")
                        last_status = current_status
                        pbar.set_description(f"Batch job {batch_id[:8]} - {current_status}")
                    
                    # Check if job is complete
                    if current_status == BatchJobStatus.COMPLETED:
                        logger.info(f"Batch job {batch_id} completed successfully")
                        return {
                            'status': current_status,
                            'output_file_id': batch_status.output_file_id,
                            'request_counts': batch_status.request_counts,
                            'batch_data': batch_status
                        }
                    elif current_status in [BatchJobStatus.FAILED, BatchJobStatus.EXPIRED, BatchJobStatus.CANCELLED]:
                        logger.error(f"Batch job {batch_id} failed with status: {current_status}")
                        return {
                            'status': current_status,
                            'error': f"Job failed with status: {current_status}",
                            'batch_data': batch_status
                        }
                    
                    # Wait before next poll
                    time.sleep(self.poll_interval)
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error polling batch job {batch_id}: {str(e)}")
                    time.sleep(self.poll_interval)
        
        # Timeout reached
        logger.error(f"Batch job {batch_id} timed out after {self.max_wait_time} seconds")
        return {
            'status': 'timeout',
            'error': f"Job timed out after {self.max_wait_time} seconds"
        }
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3)
    )
    def _download_batch_results(self, output_file_id: str, output_path: str) -> str:
        """
        Download batch job results.
        
        Args:
            output_file_id: Output file ID from completed batch job
            output_path: Local path to save results
            
        Returns:
            Path to downloaded results file
        """
        logger.info(f"Downloading batch results from file: {output_file_id}")
        
        try:
            # Get file content with extended timeout
            import httpx
            client = self.openai_client._client
            old_timeout = client.timeout
            try:
                # Extend timeout to 5 minutes for large file downloads
                client.timeout = httpx.Timeout(300.0)
                file_response = self.openai_client.files.content(output_file_id)
            finally:
                # Restore original timeout
                client.timeout = old_timeout
            
            # Save to local file
            with open(output_path, 'wb') as f:
                f.write(file_response.content)
            
            logger.info(f"Downloaded batch results to: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a gateway timeout
            if "504" in error_msg or "Gateway time-out" in error_msg:
                logger.error(f"OpenAI API gateway timeout downloading {output_file_id}. This is an OpenAI server issue.")
                logger.info("You can try again later when OpenAI's servers are less busy.")
                raise Exception(f"OpenAI gateway timeout (server issue): {output_file_id}")
            else:
                logger.error(f"Error downloading batch results: {error_msg}")
                raise
    
    def _process_batch_results(self, results_path: str, custom_id_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process downloaded batch results and prepare for indexing.
        
        Args:
            results_path: Path to downloaded results file
            custom_id_mapping: Mapping of custom IDs to original data
            
        Returns:
            List of items ready for indexing
        """
        logger.info(f"Processing batch results from: {results_path}")
        
        items_to_index = []
        successful_requests = 0
        failed_requests = 0
        
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id')
                    
                    if custom_id not in custom_id_mapping:
                        logger.warning(f"Unknown custom_id in results: {custom_id}")
                        continue
                    
                    # Check if request was successful
                    response = result.get('response')
                    if response and response.get('status_code') == 200:
                        # Extract embedding
                        body = response.get('body', {})
                        data = body.get('data', [])
                        
                        if data and len(data) > 0:
                            embedding = data[0].get('embedding')
                            if embedding:
                                # Get original item data
                                item_data = custom_id_mapping[custom_id]
                                
                                # Prepare for indexing
                                items_to_index.append({
                                    'hash_value': item_data['hash_value'],
                                    'original_string': item_data['original_string'],
                                    'field_type': item_data['field_type'],
                                    'frequency': item_data['frequency'],
                                    'vector': np.array(embedding, dtype=np.float32)
                                })
                                successful_requests += 1
                            else:
                                logger.warning(f"No embedding found for custom_id: {custom_id}")
                                failed_requests += 1
                        else:
                            logger.warning(f"No data in response for custom_id: {custom_id}")
                            failed_requests += 1
                    else:
                        # Request failed
                        error_info = result.get('error', {})
                        logger.warning(f"Failed request for custom_id {custom_id}: {error_info}")
                        failed_requests += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing result line: {str(e)}")
                    failed_requests += 1
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    failed_requests += 1
        
        logger.info(f"Processed batch results: {successful_requests} successful, {failed_requests} failed")
        return items_to_index
    
    def _process_batch_results_without_mapping(self, results_path: str) -> List[Dict[str, Any]]:
        """
        Process downloaded batch results for recovered jobs without custom_id_mapping.
        This creates a simplified mapping based on the custom_id format.
        
        Args:
            results_path: Path to downloaded results file
            
        Returns:
            List of items ready for indexing
        """
        logger.info(f"Processing batch results (recovered job) from: {results_path}")
        
        items_to_index = []
        successful_requests = 0
        failed_requests = 0
        
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id')
                    
                    if not custom_id:
                        logger.warning("No custom_id in result")
                        continue
                    
                    # Check if request was successful
                    response = result.get('response')
                    if response and response.get('status_code') == 200:
                        # Extract embedding
                        body = response.get('body', {})
                        data = body.get('data', [])
                        
                        if data and len(data) > 0:
                            embedding = data[0].get('embedding')
                            if embedding:
                                # Parse custom_id to extract info: format is "hash_field_index"
                                try:
                                    parts = custom_id.split('_')
                                    if len(parts) >= 3:
                                        hash_value = '_'.join(parts[:-2])  # Everything except last 2 parts
                                        field_type = parts[-2]
                                        index = parts[-1]
                                        
                                        # We don't have the original string, but we can still index
                                        items_to_index.append({
                                            'hash_value': hash_value,
                                            'original_string': f"Recovered string {hash_value}",  # Placeholder
                                            'field_type': field_type,
                                            'frequency': 1,  # Default frequency
                                            'vector': np.array(embedding, dtype=np.float32)
                                        })
                                        successful_requests += 1
                                    else:
                                        logger.warning(f"Cannot parse custom_id format: {custom_id}")
                                        failed_requests += 1
                                except Exception as parse_error:
                                    logger.warning(f"Error parsing custom_id {custom_id}: {parse_error}")
                                    failed_requests += 1
                            else:
                                logger.warning(f"No embedding found for custom_id: {custom_id}")
                                failed_requests += 1
                        else:
                            logger.warning(f"No data in response for custom_id: {custom_id}")
                            failed_requests += 1
                    else:
                        # Request failed
                        error_info = result.get('error', {})
                        logger.warning(f"Failed request for custom_id {custom_id}: {error_info}")
                        failed_requests += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing result line: {str(e)}")
                    failed_requests += 1
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    failed_requests += 1
        
        logger.info(f"Processed batch results (recovered): {successful_requests} successful, {failed_requests} failed")
        logger.warning("Note: Original string values not available for recovered jobs - using placeholders")
        return items_to_index
    
    def _index_embeddings_batch(self, items_to_index: List[Dict[str, Any]]) -> int:
        """
        Index a batch of embeddings in Weaviate.
        
        Args:
            items_to_index: List of items to index with vectors
            
        Returns:
            Number of successfully indexed items
        """
        logger.info(f"Indexing batch of {len(items_to_index)} items in Weaviate")
        
        indexed_count = 0
        
        try:
            # Use fixed-size batch configuration for better performance
            with self.collection.batch.fixed_size(batch_size=min(100, len(items_to_index))) as batch_writer:
                for item in items_to_index:
                    try:
                        # Generate UUID from hash value and field type for idempotency
                        uuid_input = f"{item['hash_value']}_{item['field_type']}"
                        uuid = generate_uuid5(uuid_input)
                        
                        # Remove vector from properties
                        properties = {k: v for k, v in item.items() if k != 'vector'}
                        
                        # Verify vector dimensions
                        vector_data = item['vector']
                        if len(vector_data) != self.embedding_dimensions:
                            logger.error(f"Vector dimension mismatch: {len(vector_data)} != {self.embedding_dimensions}")
                            continue
                        
                        # Add object with explicit vector format
                        batch_writer.add_object(
                            properties=properties,
                            uuid=uuid,
                            vector=vector_data.tolist()  # Convert numpy array to list
                        )
                        
                        indexed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error indexing item {item.get('hash_value', 'unknown')}: {str(e)}")
            
            logger.info(f"Successfully indexed {indexed_count}/{len(items_to_index)} items")
            
        except Exception as e:
            logger.error(f"Error in batch indexing: {str(e)}")
        
        return indexed_count
    
    def _select_strings_to_process(self, string_dict: Dict[str, str], 
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
        """
        Select strings for embedding based on field types.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            
        Returns:
            List of (hash, string, field_type, frequency) tuples to process
        """
        logger.info("Selecting strings for batch processing")
        
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
        logger.info(f"Selected {selected_count} string-field pairs for batch processing")
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
        processed_hashes_path = os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
        batch_jobs_path = os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        
        if os.path.exists(processed_hashes_path):
            try:
                with open(processed_hashes_path, 'rb') as f:
                    self.processed_hashes = set(pickle.load(f))
                logger.info(f"Loaded {len(self.processed_hashes)} processed hashes from checkpoint")
            except Exception as e:
                logger.error(f"Error loading processed hashes: {str(e)}")
                self.processed_hashes = set()
        
        if os.path.exists(batch_jobs_path):
            try:
                with open(batch_jobs_path, 'rb') as f:
                    self.batch_jobs = pickle.load(f)
                logger.info(f"Loaded {len(self.batch_jobs)} batch jobs from checkpoint")
                
                # Extract hashes from existing batch jobs to avoid reprocessing
                extracted_hashes = 0
                for job_info in self.batch_jobs.values():
                    if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                        for custom_id, item_data in job_info['file_metadata']['custom_id_mapping'].items():
                            hash_value = item_data['hash_value']
                            self.processed_hashes.add(hash_value)
                            extracted_hashes += 1
                
                if extracted_hashes > 0:
                    logger.info(f"Extracted {extracted_hashes} submitted hashes from existing batch jobs")
                    
            except Exception as e:
                logger.error(f"Error loading batch jobs: {str(e)}")
                logger.warning("Batch jobs checkpoint appears corrupted - attempting to recover")
                
                # Try to recover by querying OpenAI for existing batch jobs
                self.batch_jobs = {}
                try:
                    self._recover_batch_jobs_from_api(checkpoint_dir)
                except Exception as recovery_error:
                    logger.error(f"Failed to recover batch jobs from API: {recovery_error}")
                    logger.info("Starting with empty batch jobs - existing jobs will be rediscovered on status check")
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save checkpoint data to disk.
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save processed hashes
        processed_hashes_path = os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
        batch_jobs_path = os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        
        try:
            with open(processed_hashes_path, 'wb') as f:
                pickle.dump(list(self.processed_hashes), f)
            
            with open(batch_jobs_path, 'wb') as f:
                pickle.dump(self.batch_jobs, f)
                
            logger.info(f"Saved checkpoint: {len(self.processed_hashes)} processed hashes, {len(self.batch_jobs)} batch jobs")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def _recover_batch_jobs_from_api_readonly(self) -> None:
        """
        Read-only recovery of batch jobs from OpenAI API for status checking.
        Does not save any files or checkpoints - purely for status display.
        """
        logger.info("Attempting to recover batch jobs from OpenAI API (read-only)")
        
        try:
            recovered_jobs = 0
            after = None
            total_batches_checked = 0
            
            # Paginate through ALL batch jobs to find ours
            while True:
                # Get batch of results (up to 100 per request)
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                total_batches_checked += len(batches.data)
                
                # Process this page of results
                for batch in batches.data:
                    # Look for our batch jobs (filter by metadata or other identifiers)
                    metadata = getattr(batch, 'metadata', {})
                    if (metadata and 
                        metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        
                        # Include ALL statuses - even failed ones so we can track them
                        logger.debug(f"Found batch {batch.id}: {batch.status}")
                        self.batch_jobs[batch.id] = {
                            'batch_idx': recovered_jobs,  # Sequential numbering for recovered jobs
                            'input_file_id': batch.input_file_id,
                            'status': batch.status,
                            'created_at': batch.created_at,
                            'recovered': True,  # Mark as recovered
                            'original_description': getattr(batch, 'metadata', {}).get('description', ''),
                            'readonly_recovery': True  # Mark as read-only recovery
                        }
                        
                        if hasattr(batch, 'output_file_id') and batch.output_file_id:
                            self.batch_jobs[batch.id]['output_file_id'] = batch.output_file_id
                        
                        # For read-only recovery, assume no files are downloaded
                        self.batch_jobs[batch.id]['results_downloaded'] = False
                        self.batch_jobs[batch.id]['results_processed'] = False
                        
                        recovered_jobs += 1
                
                # Check if there are more results
                if not batches.has_more:
                    break
                    
                # Get the last batch ID for pagination
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Scanned {total_batches_checked} total batches from OpenAI API")
            if recovered_jobs > 0:
                logger.info(f"Recovered {recovered_jobs} entity resolution batch jobs (read-only)")
            else:
                logger.info("No existing entity resolution batch jobs found in OpenAI API")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI API for batch recovery: {str(e)}")
            raise

    def _recover_batch_jobs_from_api(self, checkpoint_dir: str = None) -> None:
        """
        Attempt to recover ALL batch jobs by querying OpenAI API with pagination.
        This helps when checkpoint files are corrupted.
        Also checks for existing downloaded files to update status accordingly.
        """
        logger.info("Attempting to recover ALL batch jobs from OpenAI API")
        
        try:
            recovered_jobs = 0
            after = None
            total_batches_checked = 0
            
            # Paginate through ALL batch jobs to find ours
            while True:
                # Get batch of results (up to 100 per request)
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                total_batches_checked += len(batches.data)
                
                # Process this page of results
                for batch in batches.data:
                    # Look for our batch jobs (filter by metadata or other identifiers)
                    metadata = getattr(batch, 'metadata', {})
                    if (metadata and 
                        metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        
                        # Include ALL statuses - even failed ones so we can track them
                        logger.debug(f"Recovering batch {batch.id} with input_file_id: {batch.input_file_id}")
                        self.batch_jobs[batch.id] = {
                            'batch_idx': recovered_jobs,  # Sequential numbering for recovered jobs
                            'input_file_id': batch.input_file_id,
                            'status': batch.status,
                            'created_at': batch.created_at,
                            'recovered': True,  # Mark as recovered
                            'original_description': getattr(batch, 'metadata', {}).get('description', '')
                        }
                        
                        if hasattr(batch, 'output_file_id') and batch.output_file_id:
                            self.batch_jobs[batch.id]['output_file_id'] = batch.output_file_id
                        
                        # Initialize download/processing status for all jobs
                        if batch.status == 'completed':
                            # Check for existing downloaded files if checkpoint_dir is available
                            results_downloaded = False
                            if checkpoint_dir:
                                logger.debug(f"Checking for downloaded files for job {batch.id[:8]}... in directory: {checkpoint_dir}")
                                
                                # Get all existing batch_results_*.jsonl files
                                import glob
                                existing_result_files = glob.glob(os.path.join(checkpoint_dir, 'batch_results_*.jsonl'))
                                
                                if existing_result_files:
                                    logger.debug(f"  Found {len(existing_result_files)} existing result files")
                                    # For now, mark all completed jobs as downloaded if ANY result files exist
                                    # This is a simple fix - we could get more sophisticated later
                                    results_downloaded = True
                                    logger.info(f"✅ Found existing results files - marking job {batch.id[:8]}... as downloaded")
                                else:
                                    logger.debug(f"  No result files found in checkpoint directory")
                            
                            # Set download/processing status
                            self.batch_jobs[batch.id]['results_downloaded'] = results_downloaded
                            self.batch_jobs[batch.id]['results_processed'] = False  # Conservative: assume not processed until confirmed
                        else:
                            # For non-completed jobs, these flags don't apply
                            self.batch_jobs[batch.id]['results_downloaded'] = False
                            self.batch_jobs[batch.id]['results_processed'] = False
                        
                        recovered_jobs += 1
                        
                        logger.debug(f"Recovered batch {batch.id}: {batch.status} (created: {batch.created_at})")
                
                # Check if there are more results
                if not batches.has_more:
                    break
                    
                # Get the last batch ID for pagination
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Scanned {total_batches_checked} total batches from OpenAI API")
            if recovered_jobs > 0:
                logger.info(f"Recovered {recovered_jobs} entity resolution batch jobs")
                
                # Group by status for summary
                status_counts = {}
                downloaded_count = 0
                for job_info in self.batch_jobs.values():
                    status = job_info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if job_info.get('results_downloaded', False):
                        downloaded_count += 1
                
                logger.info("Recovered job status summary:")
                for status, count in status_counts.items():
                    logger.info(f"  {status}: {count} jobs")
                
                if downloaded_count > 0:
                    logger.info(f"Found {downloaded_count} jobs with existing downloaded results")
                    
            else:
                logger.info("No existing entity resolution batch jobs found in OpenAI API")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI API for batch recovery: {str(e)}")
            raise
    
    def _estimate_tokens_for_requests(self, requests: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens needed for a list of batch requests.
        Uses tiktoken for accurate counting when available, fallback to character-based estimation.
        
        Args:
            requests: List of embedding requests
            
        Returns:
            Estimated total token count
        """
        total_tokens = 0
        
        # Initialize tokenizer if available
        encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(self.embedding_model)
            except Exception as e:
                logger.debug(f"Could not load tiktoken encoding for {self.embedding_model}: {e}")
                encoding = None
        
        for request in requests:
            # Get the input text from the request
            if 'body' in request and 'input' in request['body']:
                text = request['body']['input']
                
                if encoding is not None:
                    # Accurate token counting with tiktoken
                    try:
                        estimated_tokens = len(encoding.encode(text))
                    except Exception:
                        # Fallback to character-based estimation
                        estimated_tokens = len(text) // 4 + 10
                else:
                    # Fallback: ~4 characters per token (GPT tokenizer approximation)
                    estimated_tokens = len(text) // 4 + 10  # Add small buffer
                
                total_tokens += estimated_tokens
                
        return total_tokens
    
    def _get_current_usage(self) -> Dict[str, int]:
        """
        Get current token and request usage from active batch jobs.
        
        Returns:
            Dictionary with current usage statistics including both tokens and requests
        """
        logger.info("Checking current usage from active batch jobs...")
        
        active_jobs = 0
        estimated_active_tokens = 0
        total_active_requests = 0
        
        try:
            # Get all current batch jobs from OpenAI API
            after = None
            while True:
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                for batch in batches.data:
                    # Check if this is our job and if it's active
                    metadata = getattr(batch, 'metadata', {})
                    if (metadata and 
                        metadata.get('created_by') == 'embedding_and_indexing_batch' and
                        batch.status in ['pending', 'validating', 'in_progress', 'finalizing']):
                        
                        active_jobs += 1
                        
                        # Get actual request count and estimate tokens
                        if hasattr(batch, 'request_counts') and batch.request_counts:
                            # Use actual request count if available
                            total_requests = getattr(batch.request_counts, 'total', 0)
                            total_active_requests += total_requests
                            # Rough estimation: average 100 tokens per embedding request
                            estimated_active_tokens += total_requests * 100
                        else:
                            # Fallback estimation based on our typical batch size
                            fallback_requests = self.max_requests_per_file
                            total_active_requests += fallback_requests
                            estimated_active_tokens += fallback_requests * 100
                
                if not batches.has_more:
                    break
                    
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Found {active_jobs} active batch jobs using ~{estimated_active_tokens:,} tokens and {total_active_requests:,} requests")
            
            return {
                'active_jobs': active_jobs,
                'estimated_active_tokens': estimated_active_tokens,
                'total_active_requests': total_active_requests,
                'token_quota_limit': self.token_quota_limit,
                'request_quota_limit': self.request_quota_limit,
                'available_token_quota': max(0, self.token_quota_limit - estimated_active_tokens),
                'available_request_quota': max(0, self.request_quota_limit - total_active_requests)
            }
            
        except Exception as e:
            logger.error(f"Error checking current usage: {e}")
            # Return conservative estimates on error
            return {
                'active_jobs': 0,
                'estimated_active_tokens': 0,
                'total_active_requests': 0,
                'token_quota_limit': self.token_quota_limit,
                'request_quota_limit': self.request_quota_limit,
                'available_token_quota': self.token_quota_limit // 2,  # Conservative fallback
                'available_request_quota': self.request_quota_limit // 2  # Conservative fallback
            }
    
    def _check_quota_capacity(self, new_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if we have enough quota capacity for new batch requests.
        Validates both token limits (500M) and request limits (1M).
        
        Args:
            new_requests: List of new embedding requests to submit
            
        Returns:
            Dictionary with quota check results
        """
        # Get current usage
        usage = self._get_current_usage()
        
        # Estimate tokens needed for new requests
        estimated_new_tokens = self._estimate_tokens_for_requests(new_requests)
        new_request_count = len(new_requests)
        
        # Calculate total usage with new requests
        total_estimated_tokens = usage['estimated_active_tokens'] + estimated_new_tokens
        total_estimated_requests = usage['total_active_requests'] + new_request_count
        
        # Apply safety margins
        safe_token_limit = int(self.token_quota_limit * (1 - self.quota_safety_margin))
        safe_request_limit = int(self.request_quota_limit * (1 - self.quota_safety_margin))
        
        # Check if we're within all limits
        within_token_quota = total_estimated_tokens <= safe_token_limit
        within_request_quota = total_estimated_requests <= safe_request_limit
        within_job_limit = usage['active_jobs'] < self.max_concurrent_jobs
        
        # Ensure individual request batch doesn't exceed per-job limit
        within_job_request_limit = new_request_count <= self.max_requests_per_job
        
        can_submit = (within_token_quota and within_request_quota and 
                     within_job_limit and within_job_request_limit)
        
        result = {
            'can_submit': can_submit,
            
            # Token quota info
            'current_token_usage': usage['estimated_active_tokens'],
            'new_request_tokens': estimated_new_tokens,
            'total_estimated_tokens': total_estimated_tokens,
            'token_quota_limit': self.token_quota_limit,
            'safe_token_limit': safe_token_limit,
            'token_quota_exceeded': not within_token_quota,
            
            # Request quota info
            'current_request_usage': usage['total_active_requests'],
            'new_request_count': new_request_count,
            'total_estimated_requests': total_estimated_requests,
            'request_quota_limit': self.request_quota_limit,
            'safe_request_limit': safe_request_limit,
            'request_quota_exceeded': not within_request_quota,
            
            # Job limits
            'active_jobs': usage['active_jobs'],
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'job_limit_exceeded': not within_job_limit,
            
            # Per-job request limit
            'max_requests_per_job': self.max_requests_per_job,
            'job_request_limit_exceeded': not within_job_request_limit
        }
        
        logger.info(f"Quota check: {estimated_new_tokens:,} new tokens, {new_request_count:,} new requests, "
                   f"{usage['estimated_active_tokens']:,} active tokens, {usage['total_active_requests']:,} active requests, "
                   f"{total_estimated_tokens:,}/{safe_token_limit:,} total tokens, "
                   f"{total_estimated_requests:,}/{safe_request_limit:,} total requests "
                   f"(can_submit: {can_submit})")
        
        return result
    
    def create_batch_jobs_only(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
                             string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Create batch jobs without waiting for completion (manual polling mode).
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with job creation metrics
        """
        logger.info("Creating batch jobs for manual polling")
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # Select strings to process
        strings_to_process = self._select_strings_to_process(
            string_dict, field_hash_mapping, string_counts
        )
        
        if not strings_to_process:
            logger.info("No new strings to process")
            return {
                'status': 'no_work',
                'message': 'All eligible strings already processed',
                'elapsed_time': time.time() - start_time
            }
        
        # Split into batches if needed (due to API limits)
        batches = []
        for i in range(0, len(strings_to_process), self.max_requests_per_file):
            batch = strings_to_process[i:i+self.max_requests_per_file]
            batches.append(batch)
        
        logger.info(f"Split {len(strings_to_process)} requests into {len(batches)} batch files")
        
        jobs_created = 0
        
        # Create each batch job
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Creating batch {batch_idx + 1}/{len(batches)} with {len(batch)} requests")
            
            try:
                # Create batch requests file
                batch_file_path = os.path.join(checkpoint_dir, f'batch_requests_{batch_idx}.jsonl')
                file_metadata = self._create_batch_requests_file(batch, batch_file_path)
                
                # Upload file
                input_file_id = self._upload_batch_file(batch_file_path)
                
                # Create batch job
                # Check quota capacity before creating batch job
                batch_requests = []
                with open(batch_file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            import json
                            batch_requests.append(json.loads(line))
                
                quota_check = self._check_quota_capacity(batch_requests)
                
                if not quota_check['can_submit']:
                    logger.warning(f"🛑 Quota check failed for batch {batch_idx + 1}:")
                    logger.warning(f"  Current token usage: {quota_check['current_token_usage']:,} tokens")
                    logger.warning(f"  Current request usage: {quota_check['current_request_usage']:,} requests")
                    logger.warning(f"  New request tokens: {quota_check['new_request_tokens']:,}")
                    logger.warning(f"  New request count: {quota_check['new_request_count']:,}")
                    logger.warning(f"  Total tokens would be: {quota_check['total_estimated_tokens']:,}/{quota_check['safe_token_limit']:,}")
                    logger.warning(f"  Total requests would be: {quota_check['total_estimated_requests']:,}/{quota_check['safe_request_limit']:,}")
                    logger.warning(f"  Active jobs: {quota_check['active_jobs']}/{quota_check['max_concurrent_jobs']}")
                    
                    if quota_check['token_quota_exceeded']:
                        logger.warning(f"  ⚠️  Token quota would be exceeded")
                    if quota_check['request_quota_exceeded']:
                        logger.warning(f"  ⚠️  Request quota would be exceeded")
                    if quota_check['job_limit_exceeded']:
                        logger.warning(f"  ⚠️  Concurrent job limit would be exceeded")
                    if quota_check['job_request_limit_exceeded']:
                        logger.warning(f"  ⚠️  Per-job request limit would be exceeded ({quota_check['new_request_count']:,} > {quota_check['max_requests_per_job']:,})")
                    
                    logger.info(f"💡 Pausing batch creation after {jobs_created} jobs to respect quota limits")
                    logger.info(f"📊 Progress: {batch_idx}/{len(batches)} batches processed")
                    logger.info(f"💡 Wait for active jobs to complete, then run again to continue")
                    
                    # Clean up uploaded file since we won't use it
                    try:
                        self.openai_client.files.delete(input_file_id)
                        logger.debug(f"Cleaned up unused uploaded file {input_file_id}")
                    except Exception as cleanup_error:
                        logger.debug(f"Could not clean up file {input_file_id}: {cleanup_error}")
                    
                    # Return current progress
                    elapsed_time = time.time() - start_time
                    return {
                        'status': 'quota_paused',
                        'jobs_created': jobs_created,
                        'batches_processed': batch_idx,
                        'total_batches': len(batches),
                        'total_requests': sum(len(b) for b in batches[:batch_idx]),
                        'remaining_requests': sum(len(b) for b in batches[batch_idx:]),
                        'estimated_cost_savings': sum(len(b) for b in batches[:batch_idx]) * 0.00001 * 0.5,
                        'elapsed_time': elapsed_time,
                        'quota_info': quota_check,
                        'message': f'Created {jobs_created} batch jobs then paused due to quota limits. Wait for jobs to complete, then re-run to continue.'
                    }
                
                # Quota check passed - create batch job
                logger.info(f"✅ Quota check passed - creating batch job {batch_idx + 1}")
                # Create batch job
                batch_job_id = self._create_batch_job(
                    input_file_id, 
                    f"Entity resolution batch {batch_idx + 1}/{len(batches)}"
                )
                
                # Track batch job
                self.batch_jobs[batch_job_id] = {
                    'batch_idx': batch_idx,
                    'input_file_id': input_file_id,
                    'file_metadata': file_metadata,
                    'created_at': time.time(),
                    'status': 'submitted'
                }
                
                jobs_created += 1
                logger.info(f"Created batch job {batch_job_id} for batch {batch_idx + 1}")
                
                # Save checkpoint after each job creation
                self.save_checkpoint(checkpoint_dir)
                
            except Exception as e:
                logger.error(f"Error creating batch job for batch {batch_idx}: {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Created {jobs_created} batch jobs in {elapsed_time:.2f} seconds")
        logger.info(f"📋 Jobs will process {len(strings_to_process)} embedding requests")
        logger.info(f"⏰ Check job status manually using: python batch_manager.py --status")
        logger.info(f"📥 Download results when ready using: python batch_manager.py --download")
        
        return {
            'status': 'jobs_created',
            'jobs_created': jobs_created,
            'total_requests': len(strings_to_process),
            'estimated_cost_savings': len(strings_to_process) * 0.00001 * 0.5,  # Rough estimate
            'elapsed_time': elapsed_time,
            'message': f'Created {jobs_created} batch jobs. Use --batch-status to check progress.'
        }
    
    def check_batch_status(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Check status of all batch jobs without waiting.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Dictionary with status information for all jobs
        """
        logger.info("Checking status of batch jobs")
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # If no batch jobs found locally, try to recover from API (READ-ONLY)
        if not self.batch_jobs:
            logger.info("No local batch jobs found - attempting to recover from OpenAI API (read-only)")
            try:
                self._recover_batch_jobs_from_api_readonly()
                if self.batch_jobs:
                    logger.info(f"Recovered {len(self.batch_jobs)} batch jobs from API (read-only)")
                    # DO NOT save checkpoint during status check - keep it read-only
                    logger.debug("Status check is read-only - not saving checkpoint")
                else:
                    logger.info("No batch jobs found in OpenAI API either")
                    return {
                        'status': 'no_jobs',
                        'message': 'No batch jobs found locally or in OpenAI API.'
                    }
            except Exception as e:
                logger.error(f"Failed to recover batch jobs from API: {str(e)}")
                return {
                    'status': 'no_jobs',
                    'message': 'No batch jobs found. Create jobs first with embedding_and_indexing stage.'
                }
        
        job_statuses = {}
        # Detailed status counters for maximum granularity
        status_counts = {
            BatchJobStatus.PENDING: 0,
            BatchJobStatus.VALIDATING: 0, 
            BatchJobStatus.IN_PROGRESS: 0,
            BatchJobStatus.FINALIZING: 0,
            BatchJobStatus.COMPLETED: 0,
            BatchJobStatus.FAILED: 0,
            BatchJobStatus.EXPIRED: 0,
            BatchJobStatus.CANCELLED: 0,
            'error': 0  # For jobs we couldn't check
        }
        
        logger.info(f"Checking status of {len(self.batch_jobs)} batch jobs...")
        
        for batch_job_id, job_info in self.batch_jobs.items():
            try:
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                current_status = batch_status.status
                
                # Handle both recovered jobs (missing file_metadata) and original jobs
                request_count = None
                if 'file_metadata' in job_info:
                    request_count = job_info['file_metadata'].get('request_count')
                elif hasattr(batch_status, 'request_counts') and batch_status.request_counts:
                    # For recovered jobs, try to get request count from OpenAI batch status
                    request_count = getattr(batch_status.request_counts, 'total', None)
                
                # Enhanced status tracking with download/processing states
                download_status = 'not_applicable'
                processing_status = 'not_applicable'
                
                if current_status == BatchJobStatus.COMPLETED:
                    # Check if results have been downloaded
                    if job_info.get('results_downloaded', False):
                        download_status = 'downloaded'
                        # Check if results have been processed/indexed
                        if job_info.get('results_processed', False):
                            processing_status = 'processed'
                        else:
                            processing_status = 'pending_processing'
                    else:
                        download_status = 'pending_download'
                        processing_status = 'pending_download'
                
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info['batch_idx'],
                    'status': current_status,
                    'download_status': download_status,
                    'processing_status': processing_status,
                    'created_at': job_info['created_at'],
                    'request_count': request_count,
                    'output_file_id': getattr(batch_status, 'output_file_id', None),
                    'recovered': job_info.get('recovered', False)
                }
                
                # Update local status
                self.batch_jobs[batch_job_id]['status'] = current_status
                if hasattr(batch_status, 'output_file_id'):
                    self.batch_jobs[batch_job_id]['output_file_id'] = batch_status.output_file_id
                
                # Count detailed statuses
                if current_status in status_counts:
                    status_counts[current_status] += 1
                else:
                    # Unknown status - count as failed for backwards compatibility
                    status_counts[BatchJobStatus.FAILED] += 1
                
                batch_display = job_info.get('batch_idx', 'unknown')
                if isinstance(batch_display, int):
                    batch_display += 1
                
                # Log failure details for failed jobs
                if current_status == BatchJobStatus.FAILED:
                    error_details = ""
                    if hasattr(batch_status, 'errors') and batch_status.errors:
                        error_details = f" - Errors: {batch_status.errors}"
                    elif hasattr(batch_status, 'request_counts'):
                        counts = batch_status.request_counts
                        error_details = f" - Requests: {getattr(counts, 'total', 0)} total, {getattr(counts, 'completed', 0)} completed, {getattr(counts, 'failed', 0)} failed"
                    logger.warning(f"Job {batch_job_id[:8]}... (batch {batch_display}): {current_status}{error_details}")
                else:
                    logger.info(f"Job {batch_job_id[:8]}... (batch {batch_display}): {current_status}")
                
            except Exception as e:
                logger.error(f"Error checking job {batch_job_id}: {str(e)}")
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info.get('batch_idx', 'unknown'),
                    'status': 'error',
                    'error': str(e),
                    'recovered': job_info.get('recovered', False)
                }
                status_counts['error'] += 1
        
        # Do NOT save checkpoint during status check - keep it read-only
        logger.debug("Status check complete - no files were created or modified")
        
        # Detailed Summary
        total_jobs = len(self.batch_jobs)
        completed_count = status_counts[BatchJobStatus.COMPLETED]
        
        logger.info(f"\n📊 DETAILED BATCH JOB STATUS:")
        logger.info(f"   Total jobs: {total_jobs}")
        
        # Show each status with appropriate emoji and only if count > 0
        status_display = [
            (BatchJobStatus.PENDING, "⏳ Pending", status_counts[BatchJobStatus.PENDING]),
            (BatchJobStatus.VALIDATING, "🔍 Validating", status_counts[BatchJobStatus.VALIDATING]), 
            (BatchJobStatus.IN_PROGRESS, "🔄 In Progress", status_counts[BatchJobStatus.IN_PROGRESS]),
            (BatchJobStatus.FINALIZING, "🏁 Finalizing", status_counts[BatchJobStatus.FINALIZING]),
            (BatchJobStatus.COMPLETED, "✅ Completed", status_counts[BatchJobStatus.COMPLETED]),
            (BatchJobStatus.FAILED, "❌ Failed", status_counts[BatchJobStatus.FAILED]),
            (BatchJobStatus.EXPIRED, "⏰ Expired", status_counts[BatchJobStatus.EXPIRED]),
            (BatchJobStatus.CANCELLED, "🚫 Cancelled", status_counts[BatchJobStatus.CANCELLED]),
            ('error', "⚠️  Error", status_counts['error'])
        ]
        
        for status_key, label, count in status_display:
            if count > 0:
                logger.info(f"   {label}: {count}")
        
        return {
            'status': 'checked',
            'total_jobs': total_jobs,
            'status_counts': status_counts,  # Detailed breakdown
            'completed': completed_count,
            'job_statuses': job_statuses,
            'ready_for_download': completed_count > 0,
            # Backwards compatibility - grouped counts
            'pending': status_counts[BatchJobStatus.PENDING] + status_counts[BatchJobStatus.VALIDATING],
            'in_progress': status_counts[BatchJobStatus.IN_PROGRESS] + status_counts[BatchJobStatus.FINALIZING],
            'failed': status_counts[BatchJobStatus.FAILED] + status_counts[BatchJobStatus.EXPIRED] + 
                     status_counts[BatchJobStatus.CANCELLED] + status_counts['error']
        }
    
    def process_completed_jobs(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Download and process results from completed batch jobs.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Dictionary with processing metrics
        """
        logger.info("Processing completed batch jobs")
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        if not self.batch_jobs:
            logger.info("No batch jobs found")
            return {
                'status': 'no_jobs',
                'message': 'No batch jobs found'
            }
        
        # Find completed jobs that haven't been processed yet
        # We need to check actual status from OpenAI API, not just local cache
        completed_jobs = []
        for batch_job_id, job_info in self.batch_jobs.items():
            try:
                # Get real-time status from OpenAI API
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                current_status = batch_status.status
                
                if (current_status == BatchJobStatus.COMPLETED and 
                    not job_info.get('results_processed', False)):
                    completed_jobs.append((batch_job_id, job_info))
                    
            except Exception as e:
                logger.debug(f"Could not check status for job {batch_job_id}: {e}")
                # Fall back to cached status for this job
                if (job_info.get('status') == BatchJobStatus.COMPLETED and 
                    not job_info.get('results_processed', False)):
                    completed_jobs.append((batch_job_id, job_info))
        
        if not completed_jobs:
            logger.info("No new completed jobs to process")
            return {
                'status': 'no_completed_jobs',
                'message': 'No new completed jobs found. Check status first with --batch-status.'
            }
        
        logger.info(f"Found {len(completed_jobs)} completed jobs to process")
        
        total_processed = 0
        successful_jobs = 0
        failed_jobs = 0
        
        # Process each completed job
        for batch_job_id, job_info in completed_jobs:
            logger.info(f"Processing batch job {batch_job_id} (batch {job_info['batch_idx'] + 1})")
            
            try:
                # Get current job status to get output file ID
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                
                if not hasattr(batch_status, 'output_file_id') or not batch_status.output_file_id:
                    logger.error(f"No output file ID for job {batch_job_id}")
                    failed_jobs += 1
                    continue
                
                # Download results
                results_file_path = os.path.join(checkpoint_dir, f'batch_results_{job_info["batch_idx"]}.jsonl')
                self._download_batch_results(batch_status.output_file_id, results_file_path)
                
                # Mark results as downloaded
                self.batch_jobs[batch_job_id]['results_downloaded'] = True                
                # Process results - handle recovered jobs without file_metadata
                if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                    # Original job with full metadata
                    custom_id_mapping = job_info['file_metadata']['custom_id_mapping']
                    items_to_index = self._process_batch_results(results_file_path, custom_id_mapping)
                else:
                    # Recovered job - process without custom_id_mapping
                    logger.warning(f"Job {batch_job_id} is recovered - processing without custom_id_mapping")
                    items_to_index = self._process_batch_results_without_mapping(results_file_path)
                
                # Index in Weaviate
                indexed_count = self._index_embeddings_batch(items_to_index)
                total_processed += indexed_count
                
                # Update processed hashes
                for item in items_to_index:
                    self.processed_hashes.add(item['hash_value'])
                
                # Mark job as processed
                self.batch_jobs[batch_job_id]['results_processed'] = True
                self.batch_jobs[batch_job_id]['indexed_count'] = indexed_count
                self.batch_jobs[batch_job_id]['processed_at'] = time.time()
                
                successful_jobs += 1
                logger.info(f"✅ Processed job {batch_job_id}: indexed {indexed_count} items")
                
                # Save checkpoint after each job
                self.save_checkpoint(checkpoint_dir)
                
            except Exception as e:
                logger.error(f"Error processing job {batch_job_id}: {str(e)}")
                failed_jobs += 1
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n📊 BATCH PROCESSING COMPLETE:")
        logger.info(f"   ✅ Successful jobs: {successful_jobs}")
        logger.info(f"   ❌ Failed jobs: {failed_jobs}")
        logger.info(f"   📊 Total items indexed: {total_processed}")
        logger.info(f"   ⏱️  Processing time: {elapsed_time:.2f} seconds")
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"   🗄️  Collection now contains: {collection_count} objects")
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        return {
            'status': 'completed',
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'total_processed': total_processed,
            'elapsed_time': elapsed_time,
            'collection_count': collection_count
        }

    def process(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
              string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Process strings using batch API: create jobs, wait for completion, and index results.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with metrics about the process
        """
        logger.info("Starting batch embedding and indexing process")
        start_time = time.time()
        
        # Check if manual polling is enabled
        if self.manual_polling:
            logger.info("Manual polling mode enabled - creating jobs without waiting")
            return self.create_batch_jobs_only(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
        # Continue with automatic polling (original behavior)
        logger.info("Automatic polling mode enabled - will wait for job completion")
        
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
                'total_cost_savings': 0,
                'collection_count': collection_count
            }
        
        # Split into batches if needed (due to API limits)
        batches = []
        for i in range(0, len(strings_to_process), self.max_requests_per_file):
            batch = strings_to_process[i:i+self.max_requests_per_file]
            batches.append(batch)
        
        logger.info(f"Split {len(strings_to_process)} requests into {len(batches)} batch files")
        
        total_processed = 0
        total_jobs_created = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} requests")
            
            try:
                # Create batch requests file
                batch_file_path = os.path.join(checkpoint_dir, f'batch_requests_{batch_idx}.jsonl')
                file_metadata = self._create_batch_requests_file(batch, batch_file_path)
                
                # Upload file
                input_file_id = self._upload_batch_file(batch_file_path)
                
                # Create batch job
                batch_job_id = self._create_batch_job(
                    input_file_id, 
                    f"Entity resolution batch {batch_idx + 1}/{len(batches)}"
                )
                
                # Track batch job
                self.batch_jobs[batch_job_id] = {
                    'batch_idx': batch_idx,
                    'input_file_id': input_file_id,
                    'file_metadata': file_metadata,
                    'created_at': time.time(),
                    'status': 'created'
                }
                
                total_jobs_created += 1
                
                # Save checkpoint after creating job
                self.save_checkpoint(checkpoint_dir)
                
            except Exception as e:
                logger.error(f"Error creating batch job for batch {batch_idx}: {str(e)}")
                continue
        
        logger.info(f"Created {total_jobs_created} batch jobs, now waiting for completion...")
        
        # Wait for all jobs to complete and process results
        completed_jobs = 0
        failed_jobs = 0
        
        for batch_job_id, job_info in self.batch_jobs.items():
            if job_info.get('status') == 'completed':
                completed_jobs += 1
                continue
                
            logger.info(f"Waiting for batch job {batch_job_id} (batch {job_info['batch_idx'] + 1})")
            
            # Poll for completion
            job_result = self._poll_batch_job(batch_job_id)
            
            if job_result['status'] == BatchJobStatus.COMPLETED:
                try:
                    # Download results
                    results_file_path = os.path.join(checkpoint_dir, f'batch_results_{job_info["batch_idx"]}.jsonl')
                    self._download_batch_results(job_result['output_file_id'], results_file_path)
                    
                    # Process results - handle recovered jobs without file_metadata
                    if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                        # Original job with full metadata
                        custom_id_mapping = job_info['file_metadata']['custom_id_mapping']
                        items_to_index = self._process_batch_results(results_file_path, custom_id_mapping)
                    else:
                        # Recovered job - process without custom_id_mapping
                        logger.warning(f"Job {batch_job_id} is recovered - processing without custom_id_mapping")
                        items_to_index = self._process_batch_results_without_mapping(results_file_path)
                    
                    # Index in Weaviate
                    indexed_count = self._index_embeddings_batch(items_to_index)
                    total_processed += indexed_count
                    
                    # Update processed hashes
                    for item in items_to_index:
                        self.processed_hashes.add(item['hash_value'])
                    
                    # Update job status
                    self.batch_jobs[batch_job_id]['status'] = 'completed'
                    self.batch_jobs[batch_job_id]['indexed_count'] = indexed_count
                    
                    completed_jobs += 1
                    logger.info(f"Completed batch job {batch_job_id}: indexed {indexed_count} items")
                    
                    # Save checkpoint
                    self.save_checkpoint(checkpoint_dir)
                    
                except Exception as e:
                    logger.error(f"Error processing results for batch job {batch_job_id}: {str(e)}")
                    failed_jobs += 1
            else:
                logger.error(f"Batch job {batch_job_id} failed: {job_result.get('error', 'Unknown error')}")
                failed_jobs += 1
        
        elapsed_time = time.time() - start_time
        
        # Calculate cost savings (50% savings with batch API)
        estimated_standard_cost = len(strings_to_process) * 0.00001  # Rough estimate
        estimated_batch_cost = estimated_standard_cost * 0.5
        cost_savings = estimated_standard_cost - estimated_batch_cost
        
        logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {total_processed} strings across {completed_jobs} completed jobs")
        logger.info(f"Failed jobs: {failed_jobs}")
        logger.info(f"Estimated cost savings: ${cost_savings:.4f} (50% reduction)")
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"Collection now contains {collection_count} objects")
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        return {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'strings_processed': total_processed,
            'jobs_created': total_jobs_created,
            'jobs_completed': completed_jobs,
            'jobs_failed': failed_jobs,
            'collection_count': collection_count,
            'estimated_cost_savings': cost_savings
        }


def embedding_and_indexing_batch(config: Dict[str, Any], string_dict: Dict[str, str], 
                               field_hash_mapping: Dict[str, Dict[str, int]],
                               string_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Batch-based function for embedding generation and indexing.
    
    Args:
        config: Configuration dictionary
        string_dict: Dictionary mapping hash to string value
        field_hash_mapping: Mapping of hash to field types
        string_counts: Mapping of hash to frequency count
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Starting batch embedding and indexing process")
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize the pipeline
    pipeline = None
    metrics = {}
    
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Process the data
        metrics = pipeline.process(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
    except Exception as e:
        logger.error(f"Error in batch embedding and indexing pipeline: {str(e)}")
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


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Batch embedding and indexing for entity resolution')
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
        # Load config to get checkpoint directory
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        batch_files = [
            os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl'),
            os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        ]
        
        # Delete batch checkpoint files
        for file_path in batch_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted checkpoint file: {file_path}")
    
    # Load configuration and run batch processing
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load preprocessing data
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    try:
        # Load required data
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings")
        
        # Run batch embedding and indexing
        metrics = embedding_and_indexing_batch(config, string_dict, field_hash_mapping, string_counts)
        
        # Save metrics
        output_dir = config.get("output_dir", "data/output")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "batch_embedding_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Batch processing completed: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise