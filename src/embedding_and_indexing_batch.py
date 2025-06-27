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
        stop=stop_after_attempt(5)
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
            # Get file content
            file_response = self.openai_client.files.content(output_file_id)
            
            # Save to local file
            with open(output_path, 'wb') as f:
                f.write(file_response.content)
            
            logger.info(f"Downloaded batch results to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading batch results: {str(e)}")
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
                    self._recover_batch_jobs_from_api()
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
    
    def _recover_batch_jobs_from_api(self) -> None:
        """
        Attempt to recover ALL batch jobs by querying OpenAI API with pagination.
        This helps when checkpoint files are corrupted.
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
                for job_info in self.batch_jobs.values():
                    status = job_info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                logger.info("Recovered job status summary:")
                for status, count in status_counts.items():
                    logger.info(f"  {status}: {count} jobs")
                    
            else:
                logger.info("No existing entity resolution batch jobs found in OpenAI API")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI API for batch recovery: {str(e)}")
            raise
    
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
        logger.info(f"⏰ Check job status manually using: python main.py --batch-status")
        logger.info(f"📥 Download results when ready using: python main.py --batch-results")
        
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
        
        # If no batch jobs found locally, try to recover from API
        if not self.batch_jobs:
            logger.info("No local batch jobs found - attempting to recover from OpenAI API")
            try:
                self._recover_batch_jobs_from_api()
                if self.batch_jobs:
                    logger.info(f"Recovered {len(self.batch_jobs)} batch jobs from API")
                    # Save the recovered jobs
                    self.save_checkpoint(checkpoint_dir)
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
        pending_count = 0
        in_progress_count = 0
        completed_count = 0
        failed_count = 0
        
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
                
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info['batch_idx'],
                    'status': current_status,
                    'created_at': job_info['created_at'],
                    'request_count': request_count,
                    'output_file_id': getattr(batch_status, 'output_file_id', None),
                    'recovered': job_info.get('recovered', False)
                }
                
                # Update local status
                self.batch_jobs[batch_job_id]['status'] = current_status
                if hasattr(batch_status, 'output_file_id'):
                    self.batch_jobs[batch_job_id]['output_file_id'] = batch_status.output_file_id
                
                # Count statuses
                if current_status in [BatchJobStatus.PENDING, BatchJobStatus.VALIDATING]:
                    pending_count += 1
                elif current_status in [BatchJobStatus.IN_PROGRESS, BatchJobStatus.FINALIZING]:
                    in_progress_count += 1
                elif current_status == BatchJobStatus.COMPLETED:
                    completed_count += 1
                else:
                    failed_count += 1
                
                batch_display = job_info.get('batch_idx', 'unknown')
                if isinstance(batch_display, int):
                    batch_display += 1
                logger.info(f"Job {batch_job_id[:8]}... (batch {batch_display}): {current_status}")
                
            except Exception as e:
                logger.error(f"Error checking job {batch_job_id}: {str(e)}")
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info.get('batch_idx', 'unknown'),
                    'status': 'error',
                    'error': str(e),
                    'recovered': job_info.get('recovered', False)
                }
                failed_count += 1
        
        # Save updated checkpoint
        self.save_checkpoint(checkpoint_dir)
        
        # Summary
        total_jobs = len(self.batch_jobs)
        logger.info(f"\n📊 BATCH JOB STATUS SUMMARY:")
        logger.info(f"   Total jobs: {total_jobs}")
        logger.info(f"   ⏳ Pending: {pending_count}")
        logger.info(f"   🔄 In progress: {in_progress_count}")
        logger.info(f"   ✅ Completed: {completed_count}")
        logger.info(f"   ❌ Failed: {failed_count}")
        
        if completed_count > 0:
            logger.info(f"\n💡 {completed_count} jobs ready for processing!")
            logger.info(f"   Run: python main.py --batch-results")
        
        return {
            'status': 'checked',
            'total_jobs': total_jobs,
            'pending': pending_count,
            'in_progress': in_progress_count,
            'completed': completed_count,
            'failed': failed_count,
            'job_statuses': job_statuses,
            'ready_for_download': completed_count > 0
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
        completed_jobs = []
        for batch_job_id, job_info in self.batch_jobs.items():
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
                
                # Process results
                items_to_index = self._process_batch_results(
                    results_file_path, 
                    job_info['file_metadata']['custom_id_mapping']
                )
                
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
                    
                    # Process results
                    items_to_index = self._process_batch_results(
                        results_file_path, 
                        job_info['file_metadata']['custom_id_mapping']
                    )
                    
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