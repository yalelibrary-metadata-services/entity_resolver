#!/usr/bin/env python
"""
Knowledge Classification Embedding and Labeling System

This script processes a SKOS taxonomy in JSON-LD format, generates embeddings for concepts,
creates a Weaviate collection for storing the embeddings, and applies classification labels
to composite fields in the training dataset. It supports two classification modes:

1. Average Mode: Averages the top-k concept vectors to create a composite representation
2. Filter Mode: Groups the top-k concepts by root category, selects the winning group,
   and uses the best concept's vector from that group (not an average)

Usage:
    python classification_embedding.py --config config.yml [--reset] [--debug]
    python classification_embedding.py --diagnostic personId1 personId2 --config config.yml
    python classification_embedding.py --mode [average|filter] --config config.yml

Options:
    --config CONFIG     Path to configuration file (default: config.yml)
    --reset            Delete collections and caches to force regeneration
    --debug            Enable debug logging
    --diagnostic ID1 ID2 Compare concept vectors between two personId values
    --top_k K          Number of top concepts to consider (default: 5)
    --mode MODE        Classification mode: 'average' (vector averaging) or 'filter' (concept filtering)
                      (default: average)
    --process_ids ID1,ID2,... Process specific personIds in diagnostic mode
    --vector-mode MODE  Vector similarity mode: 'average' (default) uses averaged top concept vectors;
                      'top' uses only the top concept vector for each entity
"""

import os
import sys
import logging
import json
import pickle
import time
import argparse
import csv
import shutil
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from enum import Enum
import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# OpenAI API client for embeddings
from openai import OpenAI

# Weaviate imports
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter, MetadataQuery

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define classification modes
class ClassificationMode(str, Enum):
    AVERAGE = "average"  # Vector averaging approach
    FILTER = "filter"    # Concept filtering approach

class SKOSTaxonomyProcessor:
    """
    Processes a SKOS taxonomy in JSON-LD format and converts it to a structure
    suitable for classification.
    """
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize the SKOS taxonomy processor.
        
        Args:
            taxonomy_path: Path to the SKOS taxonomy JSON-LD file
        """
        self.taxonomy_path = taxonomy_path
        self.concept_hierarchy = {}
        self.concept_labels = {}
        self.concept_alt_labels = {}
        self.concept_paths = {}
        self.concept_definitions = {}
        self.concept_scope_notes = {}  # Added scope notes
        self.taxonomy_data = None
        self.parent_lookup = {}
        
    def load_taxonomy(self) -> Dict[str, List[str]]:
        """
        Load and process the SKOS taxonomy into a dictionary mapping domains to concepts.
        
        Returns:
            Dictionary mapping knowledge domain to list of terms
        """
        try:
            # Load the SKOS taxonomy
            with open(self.taxonomy_path, 'r') as f:
                self.taxonomy_data = json.load(f)
                
            # Extract top concepts
            top_concepts = self.taxonomy_data.get("skos:hasTopConcept", [])
            
            # Process concepts into a classification scheme
            classification_scheme = {}
            
            # Process each top concept
            for top_concept in top_concepts:
                # Get concept ID and preferred label
                concept_id = top_concept.get("@id", "")
                concept_uri = concept_id.split("/")[-1]  # Extract the last part of the URI
                
                pref_label = top_concept.get("skos:prefLabel", {}).get("@value", "")
                
                # Store concept label
                self.concept_labels[concept_uri] = pref_label
                
                # Store concept path (top level concepts have themselves as the path)
                self.concept_paths[concept_uri] = [concept_uri]
                
                # Extract concept definition if available
                if "skos:definition" in top_concept:
                    definition = top_concept["skos:definition"].get("@value", "")
                    self.concept_definitions[concept_uri] = definition
                
                # Extract scope note if available
                if "skos:scopeNote" in top_concept:
                    scope_note = top_concept["skos:scopeNote"].get("@value", "")
                    self.concept_scope_notes[concept_uri] = scope_note
                
                # Extract alternative labels if available
                if "skos:altLabel" in top_concept:
                    alt_labels = top_concept["skos:altLabel"]
                    if isinstance(alt_labels, list):
                        self.concept_alt_labels[concept_uri] = [
                            label.get("@value", "") for label in alt_labels
                        ]
                    else:
                        self.concept_alt_labels[concept_uri] = [alt_labels.get("@value", "")]
                
                # Initialize the terms list with the preferred label and alternative labels
                terms = [pref_label]
                if concept_uri in self.concept_alt_labels:
                    terms.extend(self.concept_alt_labels[concept_uri])
                
                # Add the concept to the classification scheme
                classification_scheme[concept_uri] = terms
                
                # Process narrower concepts recursively
                if "skos:narrower" in top_concept:
                    self._process_narrower_concepts(
                        top_concept["skos:narrower"], 
                        concept_uri, 
                        classification_scheme,
                        [concept_uri]  # Initial path for narrower concepts
                    )
            
            # --- FIX: Build parent lookup after processing taxonomy ---
            self.parent_lookup = self._build_parent_lookup(self.taxonomy_data)
            logger.info(f"Built parent lookup with {len(self.parent_lookup)} entries")
            
            logger.info(f"Loaded SKOS taxonomy with {len(classification_scheme)} concepts")
            return classification_scheme
            
        except Exception as e:
            logger.error(f"Error loading SKOS taxonomy: {str(e)}")
            raise
    
    def _process_narrower_concepts(self, narrower_concepts: Union[List, Dict], 
                                  parent_concept: str, 
                                  classification_scheme: Dict[str, List[str]],
                                  current_path: List[str]):
        """
        Recursively process narrower concepts in the SKOS taxonomy.
        
        Args:
            narrower_concepts: List of narrower concepts or a single concept
            parent_concept: URI of the parent concept
            classification_scheme: Dictionary to update with concepts
            current_path: Current path in the concept hierarchy
        """
        # Handle single concept case
        if not isinstance(narrower_concepts, list):
            narrower_concepts = [narrower_concepts]
        
        # Process each narrower concept
        for concept in narrower_concepts:
            # Get concept ID and preferred label
            concept_id = concept.get("@id", "")
            concept_uri = concept_id.split("/")[-1]  # Extract the last part of the URI
            
            pref_label = concept.get("skos:prefLabel", {}).get("@value", "")
            
            # Store concept label
            self.concept_labels[concept_uri] = pref_label
            
            # Update concept hierarchy
            if parent_concept not in self.concept_hierarchy:
                self.concept_hierarchy[parent_concept] = []
            self.concept_hierarchy[parent_concept].append(concept_uri)
            
            # Update concept path (append to parent path)
            new_path = current_path + [concept_uri]
            self.concept_paths[concept_uri] = new_path
            
            # Extract concept definition if available
            if "skos:definition" in concept:
                definition = concept["skos:definition"].get("@value", "")
                self.concept_definitions[concept_uri] = definition
            
            # Extract scope note if available
            if "skos:scopeNote" in concept:
                scope_note = concept["skos:scopeNote"].get("@value", "")
                self.concept_scope_notes[concept_uri] = scope_note
            
            # Extract alternative labels if available
            if "skos:altLabel" in concept:
                alt_labels = concept["skos:altLabel"]
                if isinstance(alt_labels, list):
                    self.concept_alt_labels[concept_uri] = [
                        label.get("@value", "") for label in alt_labels
                    ]
                else:
                    self.concept_alt_labels[concept_uri] = [alt_labels.get("@value", "")]
            
            # Initialize the terms list with the preferred label and alternative labels
            terms = [pref_label]
            if concept_uri in self.concept_alt_labels:
                terms.extend(self.concept_alt_labels[concept_uri])
            
            # Add the concept to the classification scheme
            classification_scheme[concept_uri] = terms
            
            # Process narrower concepts recursively
            if "skos:narrower" in concept:
                self._process_narrower_concepts(
                    concept["skos:narrower"], 
                    concept_uri, 
                    classification_scheme,
                    new_path
                )
    
    def get_concept_path(self, concept_uri: str) -> List[str]:
        """
        Get the full concept path for a given concept URI.
        
        Args:
            concept_uri: URI of the concept
            
        Returns:
            List of concept URIs representing the path from top to current concept
        """
        return self.concept_paths.get(concept_uri, [concept_uri])
    
    def get_concept_path_labels(self, concept_uri: str) -> List[str]:
        """
        Get the concept path with human-readable labels instead of URIs.
        
        Args:
            concept_uri: URI of the concept
            
        Returns:
            List of concept labels representing the path from top to current concept
        """
        path = self.get_concept_path(concept_uri)
        return [self.concept_labels.get(uri, uri) for uri in path]
    
    def get_concept_label(self, concept_uri: str) -> str:
        """
        Get the preferred label for a concept URI.
        
        Args:
            concept_uri: URI of the concept
            
        Returns:
            Preferred label for the concept
        """
        return self.concept_labels.get(concept_uri, concept_uri)
    
    def get_scope_note(self, concept_uri: str) -> str:
        """
        Get the scope note for a concept URI.
        
        Args:
            concept_uri: URI of the concept
            
        Returns:
            Scope note for the concept, or empty string if not available
        """
        return self.concept_scope_notes.get(concept_uri, "")
    
    def get_root_concept(self, concept_uri: str) -> str:
        """
        Get the root (top-level) concept for a given concept URI.
        
        Args:
            concept_uri: URI of the concept
            
        Returns:
            URI of the root concept
        """
        path = self.get_concept_path(concept_uri)
        if path and len(path) > 0:
            return path[0]
        return concept_uri

    def _build_parent_lookup(self, taxonomy_data):
        """Recursively build a mapping of child_uri -> parent_uri from skos:narrower relationships."""
        parent_lookup = {}
        def collect_parents(node, parent_uri=None):
            if isinstance(node, dict):
                if node.get('@id') and node.get('@type') == 'skos:Concept':
                    uri = node['@id'].split('/')[-1]
                    parent_short = parent_uri.split('/')[-1] if parent_uri else None
                    if uri not in parent_lookup:
                        parent_lookup[uri] = parent_short
                for k, v in node.items():
                    if k == 'skos:narrower':
                        if isinstance(v, list):
                            for child in v:
                                collect_parents(child, node.get('@id'))
                        else:
                            collect_parents(v, node.get('@id'))
                    elif isinstance(v, dict) or isinstance(v, list):
                        collect_parents(v, parent_uri)
            elif isinstance(node, list):
                for item in node:
                    collect_parents(item, parent_uri)
        collect_parents(taxonomy_data)
        return parent_lookup

class ClassificationResult:
    """
    A class to store the result of classification with multiple concepts and similarity scores.
    """
    
    def __init__(self, top_concepts: List[Dict[str, Any]], mode: ClassificationMode, concept_embeddings: Dict[str, np.ndarray]):
        """
        Initialize the classification result.
        
        Args:
            top_concepts: List of dictionaries with concept information
            mode: Classification mode used (average or filter)
            concept_embeddings: Dictionary mapping concept URIs to their embedding vectors
        """
        self.top_concepts = top_concepts
        self.mode = mode
        self.concept_embeddings = concept_embeddings
        
        # Set default values
        self.primary_concept_uri = 'unknown'
        self.primary_path = []
        self.representative_vector = None
        
        # Process based on mode
        if mode == ClassificationMode.AVERAGE:
            self._process_average_mode()
        else:  # FILTER mode
            self._process_filter_mode()
    
    def _process_average_mode(self):
        """Process classification using the averaging approach."""
        if not self.top_concepts or len(self.top_concepts) == 0:
            # Set zero vector for empty or unknown cases
            self.representative_vector = np.zeros(1536)  # Default dimension
            return
        
        # Set the primary concept to the top match
        self.primary_concept_uri = self.top_concepts[0].get('concept_uri', 'unknown')
        self.primary_path = self.top_concepts[0].get('path', [])
        
        # Average the vectors of all concepts
        concept_vectors = []
        for concept in self.top_concepts:
            concept_uri = concept.get('concept_uri')
            if concept_uri in self.concept_embeddings:
                concept_vectors.append(self.concept_embeddings[concept_uri])
        
        if concept_vectors:
            # Stack and average the vectors
            stacked_vectors = np.stack(concept_vectors)
            averaged_vector = np.mean(stacked_vectors, axis=0)
            
            # Normalize the averaged vector to unit length
            vector_norm = np.linalg.norm(averaged_vector)
            if vector_norm > 0:
                averaged_vector = averaged_vector / vector_norm
            
            self.representative_vector = averaged_vector
        else:
            # Default to zero vector if no valid concepts
            self.representative_vector = np.zeros(1536)
    
    def _process_filter_mode(self):
        """Process classification using the filtering approach."""
        if not self.top_concepts or len(self.top_concepts) == 0:
            # Set zero vector for empty or unknown cases
            self.representative_vector = np.zeros(1536)  # Default dimension
            return
        
        # Group concepts by their root category (first element in path)
        root_groups = defaultdict(list)
        for concept in self.top_concepts:
            path = concept.get('path', [])
            if path and len(path) > 0:
                root = path[0]
                root_groups[root].append(concept)
        
        # If no valid groups, use the top concept
        if not root_groups:
            self.primary_concept_uri = self.top_concepts[0].get('concept_uri', 'unknown')
            self.primary_path = self.top_concepts[0].get('path', [])
            
            # Use the vector of the top concept
            if self.primary_concept_uri in self.concept_embeddings:
                self.representative_vector = self.concept_embeddings[self.primary_concept_uri]
            else:
                self.representative_vector = np.zeros(1536)
            return
        
        # Calculate total similarity for each group
        group_scores = {}
        for root, group_concepts in root_groups.items():
            # Sum similarities for the group
            total_similarity = sum(c.get('similarity', 0) for c in group_concepts)
            group_scores[root] = total_similarity
        
        # Select the root with highest total similarity
        best_root = max(group_scores.items(), key=lambda x: x[1])[0]
        
        # Get the concepts in the best root group
        best_group = root_groups[best_root]
        
        # Sort the group by similarity (descending)
        best_group.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Set the primary concept to the highest similarity concept in the best group
        self.primary_concept_uri = best_group[0].get('concept_uri', 'unknown')
        self.primary_path = best_group[0].get('path', [])
        
        # Use the vector of the best concept directly (not an average)
        if self.primary_concept_uri in self.concept_embeddings:
            self.representative_vector = self.concept_embeddings[self.primary_concept_uri]
        else:
            self.representative_vector = np.zeros(1536)
    
    @property
    def label(self) -> str:
        """Get the primary concept label."""
        for concept in self.top_concepts:
            if concept.get('concept_uri') == self.primary_concept_uri:
                return concept.get('pref_label', 'unknown')
        return 'unknown'
    
    @property
    def path_str(self) -> str:
        """Get the primary concept path as a string."""
        if self.primary_path:
            path_labels = []
            for uri in self.primary_path:
                # Find the concept in top_concepts
                for concept in self.top_concepts:
                    if concept.get('concept_uri') == uri:
                        path_labels.append(concept.get('pref_label', uri))
                        break
                else:
                    # If not found, use the URI
                    path_labels.append(uri)
            return " > ".join(path_labels)
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the classification result to a dictionary for serialization."""
        result = {
            'top_concepts': self.top_concepts,
            'primary_concept_uri': self.primary_concept_uri,
            'primary_path': self.primary_path,
            'mode': self.mode,
            'representative_vector': (self.representative_vector.tolist() 
                                     if isinstance(self.representative_vector, np.ndarray) 
                                     else None)
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], concept_embeddings: Dict[str, np.ndarray]) -> 'ClassificationResult':
        """Create a ClassificationResult from a dictionary."""
        # Create instance with minimal initialization
        result = cls.__new__(cls)
        
        # Set attributes from data
        result.top_concepts = data['top_concepts']
        result.primary_concept_uri = data['primary_concept_uri']
        result.primary_path = data['primary_path']
        result.mode = data['mode']
        result.concept_embeddings = concept_embeddings
        
        # Convert representative_vector back to numpy array
        if data['representative_vector']:
            result.representative_vector = np.array(data['representative_vector'])
        else:
            result.representative_vector = np.zeros(1536)
        
        return result

class ClassificationEmbedding:
    """
    Manages the embedding and indexing of knowledge classification concepts
    and application of classification labels to training data.
    """
    
    def __init__(self, config: Dict[str, Any], reset_mode: bool = False):
        """
        Initialize the classification embedding processor.
        
        Args:
            config: Configuration dictionary with embedding and indexing parameters
            reset_mode: Flag indicating whether to operate in reset mode
        """
        # Store configuration
        self.config = config
        self.reset_mode = reset_mode
        
        # Configure directories
        self.input_dir = config.get("input_dir", "data/input")
        self.checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        self.output_dir = config.get("output_dir", "data/output")
        self.cache_dir = os.path.join(self.checkpoint_dir, "cache")
        
        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Default parameters
        self.top_k = config.get("top_k", 5)
        self.classification_mode = ClassificationMode(config.get("classification_mode", ClassificationMode.AVERAGE))
        
        # Configure caches
        self.embedding_cache_path = os.path.join(self.checkpoint_dir, "classification_embeddings.pkl")
        self.composite_cache_path = os.path.join(self.checkpoint_dir, "composite_embeddings.pkl")
        self.classification_cache_path = os.path.join(self.checkpoint_dir, "classification_results.pkl")
        
        # Clear caches if in reset mode
        if reset_mode:
            self._clear_caches()
        
        # OpenAI API configuration
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
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
        
        # Initialize Weaviate client
        self.weaviate_client = None
        
        # Initialize embedding mapping
        self.concept_embedding_mapping = {}
        self.composite_embedding_cache = {}
        self.classification_cache = {}
        
        # Initialize personId to classification mapping
        self.person_id_mappings = {}
        
        # Initialize SKOS taxonomy processor
        taxonomy_path = os.path.join(self.input_dir, "taxonomy_revised.json")
        self.taxonomy_processor = SKOSTaxonomyProcessor(taxonomy_path)
        
        # Connect to Weaviate
        self._init_weaviate_client()
        
        logger.info(f"Initialized ClassificationEmbedding with mode {self.classification_mode} " +
                   f"and model {self.embedding_model}" + 
                   (" in RESET mode" if reset_mode else ""))
    
    def _clear_caches(self):
        """Clear all caches and checkpoints related to classification embeddings."""
        logger.info("Clearing caches and checkpoints for fresh generation...")
        
        # List of cache files to remove
        cache_files = [
            self.embedding_cache_path,
            self.composite_cache_path,
            self.classification_cache_path,
            os.path.join(self.checkpoint_dir, "classification_embeddings.json"),
            os.path.join(self.checkpoint_dir, "person_id_mappings.pkl")
        ]
        
        # Remove each cache file
        for file_path in cache_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted cache file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {str(e)}")
        
        # Clear cache directory
        if os.path.exists(self.cache_dir):
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleared cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear cache directory {self.cache_dir}: {str(e)}")
        
        logger.info("Cache clearing complete")
    
    def _init_weaviate_client(self):
        """
        Initialize and return a Weaviate client based on configuration.
        Ensures proper connection closure on error.
        """
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
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=secure
            )
            
            # Initialize client
            client = weaviate.WeaviateClient(
                connection_params=connection_params,
                auth_client_secret=auth_client_secret
            )
            
            # Connect to Weaviate
            client.connect()
            
            logger.info(f"Connected to Weaviate at {weaviate_url}")
            self.weaviate_client = client
            
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {str(e)}")
            # Ensure no connection leaks
            if hasattr(self, 'weaviate_client') and self.weaviate_client is not None:
                try:
                    self.weaviate_client.close()
                    logger.info("Closed partially initialized Weaviate client")
                except Exception as close_error:
                    logger.error(f"Error closing partial Weaviate connection: {str(close_error)}")
            raise
    
    def load_classification_scheme(self) -> Dict[str, List[str]]:
        """
        Load the knowledge classification scheme from SKOS taxonomy.
        
        Returns:
            Dictionary mapping knowledge domains to list of concept terms
        """
        try:
            # Process the SKOS taxonomy
            classification_scheme = self.taxonomy_processor.load_taxonomy()
            logger.info(f"Loaded classification scheme with {len(classification_scheme)} concepts")
            return classification_scheme
            
        except Exception as e:
            logger.error(f"Error loading classification scheme: {str(e)}")
            raise
    
    def generate_embeddings(self, classification_scheme: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all concepts in the classification scheme.
        Uses parallel processing for efficiency.
        
        Args:
            classification_scheme: Dictionary mapping concept URI to list of terms
            
        Returns:
            Dictionary mapping concept URI to embedding vector
        """
        logger.info("Generating embeddings for classification scheme concepts")
        
        # Prepare items to embed
        items_to_embed = []
        for concept_uri, terms in classification_scheme.items():
            pref_label = terms[0]  # First term is the preferred label
            
            # Use hierarchical embedding string for maximum context
            embedding_text = self.build_hierarchical_embedding_string(
                self.taxonomy_processor.taxonomy_data,
                concept_uri,
                self.taxonomy_processor.concept_labels,
                self.taxonomy_processor.concept_definitions,
                self.taxonomy_processor.concept_scope_notes,
                self.taxonomy_processor.concept_alt_labels,
                self.taxonomy_processor.parent_lookup
            )
            
            # Add to embedding list
            items_to_embed.append((concept_uri, pref_label, embedding_text))
        
        # Calculate number of workers based on available CPUs
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        max_workers = min(available_cores, self.config.get("embedding_workers", 4))
        
        logger.info(f"Embedding {len(items_to_embed)} concepts using {max_workers} workers")
        
        # Create lock for synchronizing counter updates
        lock = threading.Lock()
        
        # Reset rate limits
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Prepare batches
        batches = []
        for i in range(0, len(items_to_embed), self.batch_size):
            batch = items_to_embed[i:i+self.batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        embedding_mapping = {}
        total_tokens = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing
            futures = {executor.submit(self._process_embedding_batch, batch, lock): i 
                     for i, batch in enumerate(batches)}
            
            # Process results as they complete with progress visualization
            with tqdm(total=len(batches), desc="Generating embeddings", unit="batch") as pbar:
                for future in as_completed(futures):
                    batch_embeddings, tokens_used = future.result()
                    embedding_mapping.update(batch_embeddings)
                    total_tokens += tokens_used
                    pbar.update(1)
                    pbar.set_postfix({
                        "concepts": len(embedding_mapping), 
                        "tokens": total_tokens
                    })
        
        logger.info(f"Generated embeddings for {len(embedding_mapping)} concepts using {total_tokens} tokens")
        self.concept_embedding_mapping = embedding_mapping
        return embedding_mapping
    
    def save_embeddings(self) -> str:
        """
        Save embeddings to mapping files in the checkpoint directory.
        
        Returns:
            Path to the saved concept mapping file
        """
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save concept embedding mapping
        mapping_path = self.embedding_cache_path
        
        try:
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.concept_embedding_mapping, f)
                
            logger.info(f"Saved {len(self.concept_embedding_mapping)} concept embeddings to {mapping_path}")
            
            # Also save in JSON format for potential inspection
            json_path = os.path.join(self.checkpoint_dir, "classification_embeddings.json")
            with open(json_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_mapping = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in self.concept_embedding_mapping.items()}
                json.dump(json_mapping, f)
            
            # Save composite embedding cache if not empty
            if self.composite_embedding_cache:
                composite_path = self.composite_cache_path
                with open(composite_path, 'wb') as f:
                    pickle.dump(self.composite_embedding_cache, f)
                logger.info(f"Saved {len(self.composite_embedding_cache)} composite embeddings to {composite_path}")
            
            # Save classification cache if not empty
            if self.classification_cache:
                # Convert ClassificationResult objects to dictionaries
                serializable_cache = {}
                for key, value in self.classification_cache.items():
                    serializable_cache[key] = value.to_dict()
                
                with open(self.classification_cache_path, 'wb') as f:
                    pickle.dump(serializable_cache, f)
                logger.info(f"Saved {len(self.classification_cache)} classification results to {self.classification_cache_path}")
            
            # Save person ID mappings if not empty
            if self.person_id_mappings:
                person_id_path = os.path.join(self.checkpoint_dir, "person_id_mappings.pkl")
                with open(person_id_path, 'wb') as f:
                    pickle.dump(self.person_id_mappings, f)
                logger.info(f"Saved {len(self.person_id_mappings)} person ID mappings to {person_id_path}")
            
            return mapping_path
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load embeddings from mapping files.
        Skip loading if in reset mode.
        
        Returns:
            Dictionary mapping concept URI to embedding vector
        """
        # Skip loading embeddings if in reset mode
        if self.reset_mode:
            logger.info("Reset mode: Skipping embedding cache loading")
            return {}
            
        mapping_path = self.embedding_cache_path
        
        try:
            # Load concept embeddings
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    embedding_mapping = pickle.load(f)
                logger.info(f"Loaded {len(embedding_mapping)} concept embeddings from {mapping_path}")
                self.concept_embedding_mapping = embedding_mapping
            else:
                logger.warning(f"Concept embedding mapping file not found at {mapping_path}")
                self.concept_embedding_mapping = {}
            
            # Load composite embeddings if available
            composite_path = self.composite_cache_path
            if os.path.exists(composite_path):
                with open(composite_path, 'rb') as f:
                    composite_mapping = pickle.load(f)
                logger.info(f"Loaded {len(composite_mapping)} composite embeddings from {composite_path}")
                self.composite_embedding_cache = composite_mapping
            else:
                logger.info("No composite embedding cache found, starting with empty cache")
                self.composite_embedding_cache = {}
            
            # Load classification cache if available
            if os.path.exists(self.classification_cache_path):
                with open(self.classification_cache_path, 'rb') as f:
                    serializable_cache = pickle.load(f)
                
                # Convert dictionaries back to ClassificationResult objects
                self.classification_cache = {}
                for key, value in serializable_cache.items():
                    self.classification_cache[key] = ClassificationResult.from_dict(value, self.concept_embedding_mapping)
                
                logger.info(f"Loaded {len(self.classification_cache)} classification results from {self.classification_cache_path}")
            else:
                logger.info("No classification cache found, starting with empty cache")
                self.classification_cache = {}
            
            # Load person ID mappings if available
            person_id_path = os.path.join(self.checkpoint_dir, "person_id_mappings.pkl")
            if os.path.exists(person_id_path):
                with open(person_id_path, 'rb') as f:
                    self.person_id_mappings = pickle.load(f)
                logger.info(f"Loaded {len(self.person_id_mappings)} person ID mappings from {person_id_path}")
            else:
                self.person_id_mappings = {}
            
            return self.concept_embedding_mapping
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return {}
    
    def index_embeddings(self, classification_scheme: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Index embeddings in Weaviate.
        Creates a ConceptScheme collection and indexes all concepts with their embeddings.
        
        Args:
            classification_scheme: Dictionary mapping concept URI to list of terms
            
        Returns:
            Dictionary with indexing metrics
        """
        logger.info("Indexing classification scheme embeddings in Weaviate")
        
        # Create schema - will handle reset mode appropriately
        collection = self._create_classification_schema()
        
        # Prepare items to index
        items_to_index = []
        for concept_uri, terms in classification_scheme.items():
            # Skip if we have no embedding for this concept
            if concept_uri not in self.concept_embedding_mapping:
                logger.warning(f"No embedding found for concept '{concept_uri}'")
                continue
                
            # Get preferred label and alternative labels
            pref_label = terms[0]
            alt_labels = terms[1:] if len(terms) > 1 else []
            
            # Get concept definition if available
            definition = self.taxonomy_processor.concept_definitions.get(concept_uri, "")
            
            # Get scope note if available
            if concept_uri in self.taxonomy_processor.concept_scope_notes:
                scope_note = self.taxonomy_processor.concept_scope_notes[concept_uri]
            else:
                scope_note = ""
            
            # Get concept path if available
            path = self.taxonomy_processor.get_concept_path_labels(concept_uri)
            
            # Add item to indexing list
            embedding_string = self.build_hierarchical_embedding_string(
                self.taxonomy_processor.taxonomy_data,
                concept_uri,
                self.taxonomy_processor.concept_labels,
                self.taxonomy_processor.concept_definitions,
                self.taxonomy_processor.concept_scope_notes,
                self.taxonomy_processor.concept_alt_labels,
                self.taxonomy_processor.parent_lookup
            )
            print(f"DEBUG: concept_uri={concept_uri} embedding_string={embedding_string}")
            items_to_index.append({
                'concept_uri': concept_uri,
                'pref_label': pref_label,
                'alt_labels': alt_labels,
                'definition': definition,
                'scope_note': scope_note,
                'path': path,
                'vector': self.concept_embedding_mapping[concept_uri],
                'embedding_string': embedding_string
            })
        
        logger.info(f"Prepared {len(items_to_index)} items to index")
        
        # Index items in batches
        batch_size = self.config.get("weaviate_batch_size", 100)
        indexed_count = 0
        
        with tqdm(total=len(items_to_index), desc="Indexing items", unit="item") as pbar:
            for i in range(0, len(items_to_index), batch_size):
                batch = items_to_index[i:i+batch_size]
                
                try:
                    with collection.batch.fixed_size(batch_size=min(100, len(batch))) as batch_writer:
                        for item in batch:
                            # Robust vector validation and logging
                            vector = item.get('vector', None)
                            #logger.debug(f"{vector}")
                            if vector is None:
                                logger.debug(f"[INDEX] No vector found for {item.get('concept_uri', 'unknown')}. Skipping.")
                                continue
                            if not isinstance(vector, (list, tuple)) and not hasattr(vector, 'tolist'):
                                logger.debug(f"[INDEX] Vector for {item.get('concept_uri', 'unknown')} is not list/array: type={type(vector)}, value={vector}. Skipping.")
                                continue
                            if hasattr(vector, 'tolist'):
                                vector = vector.tolist()
                            if not isinstance(vector, list) or len(vector) == 0 or not all(isinstance(x, (int, float)) for x in vector):
                                logger.debug(f"[INDEX] Vector for {item.get('concept_uri', 'unknown')} is invalid or empty: {vector}. Skipping.")
                                continue
                            logger.debug(f"[INDEX] Vector for {item.get('concept_uri', 'unknown')}: type={type(vector)}, len={len(vector)}")

                            # Generate UUID from concept URI for idempotency
                            uuid = generate_uuid5(item['concept_uri'])
                            
                            # Remove vector from properties
                            properties = {k: v for k, v in item.items() if k != 'vector'}
                            
                            # Add object with vector
                            batch_writer.add_object(
                                properties=properties,
                                uuid=uuid,
                                vector=vector
                            )
                            
                            indexed_count += 1
                            
                        # Update progress bar
                        pbar.update(len(batch))
                        
                except Exception as e:
                    logger.error(f"Error in batch indexing: {str(e)}")
        
        # Get final collection stats
        try:
            result = collection.aggregate.over_all(
                total_count=True
            )
            collection_count = result.total_count
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = indexed_count
        
        logger.info(f"Indexed {indexed_count} items, collection has {collection_count} objects")
        
        # Return metrics
        metrics = {
            'status': 'completed',
            'indexed_count': indexed_count,
            'collection_count': collection_count
        }
        
        return metrics
    
    def classify_composite_field(self, composite_text: str, composite_hash: str = None, 
                               top_k: int = None, mode: ClassificationMode = None) -> ClassificationResult:
        """
        Classify a composite field by finding the top K most similar concepts.
        Implements caching to avoid repeated API calls for identical text.
        
        Args:
            composite_text: Composite field text to classify
            composite_hash: Optional pre-computed hash of the composite text
            top_k: Number of top concepts to consider (default: self.top_k)
            mode: Classification mode to use (default: self.classification_mode)
            
        Returns:
            ClassificationResult object containing classification info
        """
        # Use defaults if not specified
        if top_k is None:
            top_k = self.top_k
        
        if mode is None:
            mode = self.classification_mode
        
        try:
            # Skip empty text
            if not composite_text or composite_text == 'NULL':
                # Return empty classification result
                return ClassificationResult(
                    top_concepts=[{
                        'concept_uri': 'unknown',
                        'pref_label': 'unknown',
                        'path': [],
                        'similarity': 0.0
                    }],
                    mode=mode,
                    concept_embeddings=self.concept_embedding_mapping
                )
            
            # Generate hash if not provided
            if composite_hash is None:
                composite_hash = hashlib.md5(composite_text.encode('utf-8')).hexdigest()
            
            # Check classification cache first - with the specific top_k and mode values
            cache_key = f"{composite_hash}_{top_k}_{mode}"
            if cache_key in self.classification_cache:
                return self.classification_cache[cache_key]
            
            # Get vector embedding for composite text
            # Check if we already have an embedding for this composite
            if composite_hash in self.composite_embedding_cache:
                vector = self.composite_embedding_cache[composite_hash]
            else:
                # Only generate embedding if not in cache
                embedding, _ = self._get_embeddings_batch([composite_text])
                vector = embedding[0]
                # Cache the embedding
                self.composite_embedding_cache[composite_hash] = vector
            
            # Get ConceptScheme collection
            collection = self.weaviate_client.collections.get("ConceptScheme")
            
            # Execute near_vector search for top K concepts
            result = collection.query.near_vector(
                near_vector=vector.tolist(),
                limit=top_k,  # Get top K results
                return_properties=["concept_uri", "pref_label", "path", "scope_note", "definition"],
                return_metadata=MetadataQuery(distance=True)
            )
            
            # Process results
            if result.objects and len(result.objects) > 0:
                # Extract concept information
                concepts = []
                
                for obj in result.objects:
                    # Get concept uri
                    concept_uri = obj.properties.get('concept_uri')
                    if not concept_uri:
                        continue
                        
                    # Calculate similarity from distance
                    distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0.0
                    similarity = 1.0 - distance
                    
                    # Create concept info dictionary
                    concept_info = {
                        'concept_uri': concept_uri,
                        'pref_label': obj.properties.get('pref_label', ''),
                        'path': obj.properties.get('path', []),
                        'similarity': similarity,
                        'scope_note': obj.properties.get('scope_note', ''),
                        'definition': obj.properties.get('definition', '')
                    }
                    
                    concepts.append(concept_info)
                
                # Create ClassificationResult based on the specified mode
                if concepts:
                    result = ClassificationResult(concepts, mode, self.concept_embedding_mapping)
                    self.classification_cache[cache_key] = result
                    return result
            
            # If we get here, no valid classifications were found
            # Return unknown classification
            unknown_result = ClassificationResult(
                top_concepts=[{
                    'concept_uri': 'unknown',
                    'pref_label': 'unknown',
                    'path': [],
                    'similarity': 0.0
                }],
                mode=mode,
                concept_embeddings=self.concept_embedding_mapping
            )
            
            # Cache the result
            self.classification_cache[cache_key] = unknown_result
            
            return unknown_result
            
        except Exception as e:
            logger.error(f"Error classifying composite field: {str(e)}")
            # Return empty classification on error
            return ClassificationResult(
                top_concepts=[{
                    'concept_uri': 'unknown',
                    'pref_label': 'unknown',
                    'path': [],
                    'similarity': 0.0
                }],
                mode=mode,
                concept_embeddings=self.concept_embedding_mapping
            )
    
    def _get_embeddings_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        """
        Get embeddings for a batch of texts from the OpenAI API.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_count) where embeddings is a list of numpy arrays
            and token_count is the total tokens used
        """
        try:
            # Get embeddings using the client
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            # Extract embeddings
            embeddings = [np.array(embedding_data.embedding, dtype=np.float32) 
                        for embedding_data in response.data]
            
            # Get token count
            token_count = response.usage.total_tokens
            
            logger.debug(f"Generated {len(embeddings)} embeddings using {token_count} tokens")
            return embeddings, token_count
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _process_embedding_batch(self, items: List[Tuple[str, str, str]],
                               lock: threading.Lock) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Process a batch of items: generate embeddings for concepts.
        
        Args:
            items: List of (concept_uri, pref_label, embedding_text) tuples
            lock: Lock for thread-safe counter updates
            
        Returns:
            Tuple of (embeddings_dict, tokens_used)
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
        
        # Extract texts to embed - use the embedding_text which contains combined terms
        texts = [item[2] for item in items]
        
        try:
            # Get embeddings
            embeddings, token_count = self._get_embeddings_batch(texts)
            
            # Update rate limiting counters
            with lock:
                self.requests_this_minute += 1
                self.tokens_this_minute += token_count
            
            # Create dictionary mapping concept_uri to embedding
            embeddings_dict = {}
            for i, (concept_uri, pref_label, _) in enumerate(items):
                embeddings_dict[concept_uri] = embeddings[i]
            
            return embeddings_dict, token_count
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {}, 0
    
    def process_specific_person_id(self, person_id: str, training_dataset_path: str = None):
        """
        Process a specific person ID for diagnostic purposes.
        
        Args:
            person_id: The person ID to process
            training_dataset_path: Optional path to the training dataset (default: from config)
            
        Returns:
            Dictionary with processed data or error message
        """
        logger.info(f"Processing specific person ID: {person_id}")
        
        # Check if already processed
        if person_id in self.person_id_mappings:
            logger.info(f"Person ID {person_id} already processed")
            return {
                'status': 'success',
                'person_id': person_id,
                'data': self.person_id_mappings[person_id]
            }
        
        # Determine training dataset path
        if not training_dataset_path:
            training_dataset_path = os.path.join(self.input_dir, "training_dataset.csv")
        
        if not os.path.exists(training_dataset_path):
            logger.error(f"Training dataset not found at {training_dataset_path}")
            return {'status': 'error', 'message': f'Training dataset not found at {training_dataset_path}'}
        
        try:
            # Detect delimiter
            def detect_delimiter(file_path, num_lines=5):
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = ''.join([f.readline() for _ in range(num_lines)])
                
                # Count common delimiters
                delimiters = [',', '\t', ';', '|']
                counts = {d: sample.count(d) for d in delimiters}
                
                # Return the most common delimiter
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                return most_common
            
            delimiter = detect_delimiter(training_dataset_path)
            
            # Define hash function
            def hash_string(string_val):
                """Generate a hash for a string value."""
                if not string_val or string_val == 'NULL':
                    return 'NULL'
                return hashlib.md5(string_val.encode('utf-8')).hexdigest()
            
            # Read the dataset and find the person ID
            with open(training_dataset_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    if row.get('personId') == person_id:
                        # Found the person ID
                        composite = row.get('composite', '')
                        
                        if not composite or composite == 'NULL':
                            logger.error(f"Person ID {person_id} has no composite field")
                            return {'status': 'error', 'message': f'Person ID {person_id} has no composite field'}
                        
                        # Generate hash
                        composite_hash = hash_string(composite)
                        
                        # Map person ID to composite
                        self.person_id_mappings[person_id] = {
                            'composite': composite,
                            'composite_hash': composite_hash
                        }
                        
                        # Create composite embedding if needed
                        if composite_hash not in self.composite_embedding_cache:
                            embedding, _ = self._get_embeddings_batch([composite])
                            self.composite_embedding_cache[composite_hash] = embedding[0]
                            logger.info(f"Generated embedding for person ID {person_id}")
                        
                        # Classify in both modes
                        for mode in ClassificationMode:
                            cache_key = f"{composite_hash}_{self.top_k}_{mode}"
                            if cache_key not in self.classification_cache:
                                self.classify_composite_field(composite, composite_hash, self.top_k, mode)
                        
                        # Save caches for future use
                        self.save_embeddings()
                        
                        logger.info(f"Successfully processed person ID {person_id}")
                        return {
                            'status': 'success',
                            'person_id': person_id,
                            'data': self.person_id_mappings[person_id]
                        }
            
            # If we get here, person ID not found
            logger.error(f"Person ID {person_id} not found in dataset")
            return {'status': 'error', 'message': f'Person ID {person_id} not found in dataset'}
            
        except Exception as e:
            logger.error(f"Error processing person ID {person_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def compare_person_ids(self, person_id1: str, person_id2: str, mode: ClassificationMode = None, vector_mode: str = None) -> Dict[str, Any]:
        """
        Compare the concept vectors of two person IDs.
        
        Args:
            person_id1: First person ID
            person_id2: Second person ID
            mode: Classification mode to use (default: self.classification_mode)
            vector_mode: Vector similarity mode to use (default: 'average')
            
        Returns:
            Dictionary with comparison results
        """
        # Use default mode if not specified
        if mode is None:
            mode = self.classification_mode
        
        # Use default vector mode if not specified
        if vector_mode is None:
            vector_mode = 'average'
        
        logger.info(f"Comparing person IDs: {person_id1} vs {person_id2} using {mode} mode and {vector_mode} vector mode")
        
        # Process person IDs if not already in mappings
        if person_id1 not in self.person_id_mappings:
            result = self.process_specific_person_id(person_id1)
            if result['status'] != 'success':
                return {'status': 'error', 'message': f"Failed to process person ID {person_id1}: {result['message']}"}
        
        if person_id2 not in self.person_id_mappings:
            result = self.process_specific_person_id(person_id2)
            if result['status'] != 'success':
                return {'status': 'error', 'message': f"Failed to process person ID {person_id2}: {result['message']}"}
        
        # Get the composite hashes for these person IDs
        composite_hash1 = self.person_id_mappings[person_id1].get('composite_hash')
        composite_hash2 = self.person_id_mappings[person_id2].get('composite_hash')
        
        if not composite_hash1 or not composite_hash2:
            logger.error(f"Missing composite hash for one or both person IDs")
            return {'status': 'error', 'message': 'Missing composite hash for one or both person IDs'}
        
        # Get classification results for both person IDs
        cache_key1 = f"{composite_hash1}_{self.top_k}_{mode}"
        cache_key2 = f"{composite_hash2}_{self.top_k}_{mode}"
        
        if cache_key1 not in self.classification_cache or cache_key2 not in self.classification_cache:
            logger.error(f"Missing classification for one or both person IDs in {mode} mode")
            return {'status': 'error', 'message': f'Missing classification for one or both person IDs in {mode} mode'}
        
        classification1 = self.classification_cache[cache_key1]
        classification2 = self.classification_cache[cache_key2]
        
        # Get the composite strings
        composite1 = self.person_id_mappings[person_id1].get('composite', 'Unknown')
        composite2 = self.person_id_mappings[person_id2].get('composite', 'Unknown')
        
        # Calculate similarity between representative vectors or top concept vectors
        if vector_mode == 'top':
            # Use only the top concept vector for each entity
            def get_top_vector(classification):
                if classification.top_concepts and classification.top_concepts[0].get('concept_uri') in classification.concept_embeddings:
                    return classification.concept_embeddings[classification.top_concepts[0]['concept_uri']]
                return np.zeros(1536)
            vector1 = get_top_vector(classification1)
            vector2 = get_top_vector(classification2)
            vector_similarity = self.calculate_vector_similarity(vector1, vector2)
            similarity_mode = 'top'
        else:
            # Default: use representative vectors (average mode)
            vector_similarity = self.calculate_vector_similarity(
                classification1.representative_vector, 
                classification2.representative_vector
            )
            similarity_mode = 'average'
        
        # Prepare comparison result
        comparison = {
            'status': 'success',
            'mode': mode,
            'vector_similarity': vector_similarity,
            'vector_similarity_mode': similarity_mode,
            'person_id1': {
                'id': person_id1,
                'composite': composite1,
                'primary_concept_label': classification1.label,
                'primary_concept_path': classification1.path_str,
                'top_concepts': classification1.top_concepts[:3]  # Include just top 3 for brevity
            },
            'person_id2': {
                'id': person_id2,
                'composite': composite2,
                'primary_concept_label': classification2.label,
                'primary_concept_path': classification2.path_str,
                'top_concepts': classification2.top_concepts[:3]  # Include just top 3 for brevity
            }
        }
        
        return comparison
    
    def calculate_vector_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity value (0-1)
        """
        # Handle zero vectors
        if np.all(vector1 == 0) or np.all(vector2 == 0):
            return 0.0
        
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vector1_normalized = vector1 / norm1
        vector2_normalized = vector2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(vector1_normalized, vector2_normalized)
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, similarity))
    
    def update_training_dataset(self) -> Dict[str, Any]:
        """
        Update training dataset with classification labels.
        Uses the full composite string for classification with optimized batch processing.
        
        Returns:
            Dictionary with update metrics
        """
        logger.info(f"Updating training dataset with classification labels using {self.classification_mode} mode")
        
        # Load training dataset
        training_dataset_path = os.path.join(self.input_dir, "training_dataset.csv")
        
        if not os.path.exists(training_dataset_path):
            logger.error(f"Training dataset not found at {training_dataset_path}")
            return {'status': 'error', 'message': 'Training dataset not found'}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Output path for updated dataset
        output_path = os.path.join(self.output_dir, "training_dataset_classified.csv")
        
        try:
            # Detect CSV delimiter by analyzing the first few lines
            def detect_delimiter(file_path, num_lines=5):
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = ''.join([f.readline() for _ in range(num_lines)])
                
                # Count common delimiters
                delimiters = [',', '\t', ';', '|']
                counts = {d: sample.count(d) for d in delimiters}
                
                # Return the most common delimiter
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                return most_common
            
            delimiter = detect_delimiter(training_dataset_path)
            
            # Define a hash function for string values
            def hash_string(string_val):
                """Generate a hash for a string value."""
                if not string_val or string_val == 'NULL':
                    return 'NULL'
                return hashlib.md5(string_val.encode('utf-8')).hexdigest()
            
            # First pass: collect all unique composite values
            unique_composites = {}
            person_id_to_composite = {}
            
            with open(training_dataset_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    composite = row.get('composite', '')
                    person_id = row.get('personId', '')
                    
                    if composite and composite != 'NULL':
                        composite_hash = hash_string(composite)
                        unique_composites[composite_hash] = composite
                        
                        # Map person ID to composite
                        if person_id:
                            person_id_to_composite[person_id] = {
                                'composite': composite,
                                'composite_hash': composite_hash
                            }
            
            # Update person ID mappings
            self.person_id_mappings.update(person_id_to_composite)
            
            logger.info(f"Found {len(unique_composites)} unique composite fields to classify")
            
            # Batch process all unique composites that aren't already in caches
            new_composites = {h: v for h, v in unique_composites.items() 
                             if h not in self.composite_embedding_cache}
            
            if new_composites:
                logger.info(f"Batch processing {len(new_composites)} new composite fields")
                # Convert to lists for processing
                hashes = list(new_composites.keys())
                texts = list(new_composites.values())
                
                # Process in manageable batches
                batch_size = self.batch_size
                processed = 0
                
                with tqdm(total=len(new_composites), desc="Generating composite embeddings", unit="field") as pbar:
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        batch_hashes = hashes[i:i+batch_size]
                        
                        # Generate embeddings for this batch
                        try:
                            embeddings, _ = self._get_embeddings_batch(batch_texts)
                            
                            # Cache embeddings
                            for j, hash_val in enumerate(batch_hashes):
                                self.composite_embedding_cache[hash_val] = embeddings[j]
                            
                            processed += len(batch_texts)
                            pbar.update(len(batch_texts))
                            
                        except Exception as e:
                            logger.error(f"Error batch processing composite fields: {str(e)}")
                
                logger.info(f"Successfully processed {processed}/{len(new_composites)} new composite fields")
            
            # Now classify each composite field in the current mode
            logger.info(f"Classifying composite fields using cached embeddings (top K: {self.top_k}, mode: {self.classification_mode})")
            
            with tqdm(total=len(unique_composites), desc="Classifying unique composites", unit="field") as pbar:
                for composite_hash, composite_text in unique_composites.items():
                    cache_key = f"{composite_hash}_{self.top_k}_{self.classification_mode}"
                    if cache_key not in self.classification_cache:
                        self.classify_composite_field(composite_text, composite_hash, self.top_k, self.classification_mode)
                    pbar.update(1)
            
            # Now read and write the CSV with classifications
            with open(training_dataset_path, 'r', newline='', encoding='utf-8') as f, \
                 open(output_path, 'w', newline='', encoding='utf-8') as g:
                
                # Read headers with detected delimiter
                reader = csv.DictReader(f, delimiter=delimiter)
                fieldnames = list(reader.fieldnames) + [
                    'classification_label', 
                    'classification_path',
                    'top_concepts_json',
                    'classification_mode',
                    'vector_hash',
                    'child1_embedding_string',
                    'child2_embedding_string'
                ]
                
                # Create writer with same delimiter for consistency
                writer = csv.DictWriter(g, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                
                # Process each row - should be fast now with everything cached
                updated_count = 0
                classified_count = 0
                total_rows = 0
                
                for row in tqdm(reader, desc="Writing classified dataset", unit="row"):
                    total_rows += 1
                    
                    # Get the composite string and hash
                    composite = row.get('composite', '')
                    composite_hash = hash_string(composite)
                    
                    # Get classification from cache
                    classification_label = "unknown"
                    classification_path = ""
                    top_concepts_json = "[]"
                    vector_hash = ""
                    child1_embedding_string = ""
                    child2_embedding_string = ""
                    
                    cache_key = f"{composite_hash}_{self.top_k}_{self.classification_mode}"
                    if cache_key in self.classification_cache:
                        classification_result = self.classification_cache[cache_key]
                        
                        if classification_result.primary_concept_uri != "unknown":
                            # Get human-readable label for the concept
                            classification_label = classification_result.label
                            # Join path with ' > ' for readability
                            classification_path = classification_result.path_str
                            # Store top concepts as JSON
                            top_concepts_json = json.dumps([
                                {
                                    'concept_uri': c.get('concept_uri'),
                                    'pref_label': c.get('pref_label'),
                                    'similarity': c.get('similarity')
                                }
                                for c in classification_result.top_concepts
                            ])
                            # Generate a hash of the representative vector for comparison
                            if isinstance(classification_result.representative_vector, np.ndarray):
                                # Convert to bytes, then hash
                                vector_bytes = classification_result.representative_vector.tobytes()
                                vector_hash = hashlib.md5(vector_bytes).hexdigest()
                            
                            # --- Populate child1_embedding_string and child2_embedding_string ---
                            taxonomy_data = self.taxonomy_processor.taxonomy_data
                            labels = self.taxonomy_processor.concept_labels
                            definitions = self.taxonomy_processor.concept_definitions
                            scope_notes = self.taxonomy_processor.concept_scope_notes
                            alt_labels = self.taxonomy_processor.concept_alt_labels
                            parent_lookup = self.taxonomy_processor.parent_lookup
                            # child1 = primary concept
                            if hasattr(classification_result, 'primary_concept_uri'):
                                child1_embedding_string = self.build_hierarchical_embedding_string(
                                    taxonomy_data,
                                    classification_result.primary_concept_uri,
                                    labels, definitions, scope_notes, alt_labels, parent_lookup
                                )
                            # child2 = 2nd concept in top_concepts (if available)
                            if hasattr(classification_result, 'top_concepts') and len(classification_result.top_concepts) > 1:
                                child2_uri = classification_result.top_concepts[1].get('concept_uri')
                                if child2_uri:
                                    child2_embedding_string = self.build_hierarchical_embedding_string(
                                        taxonomy_data,
                                        child2_uri,
                                        labels, definitions, scope_notes, alt_labels, parent_lookup
                                    )
                        
                            classified_count += 1
                    
                    # Add classification fields to row
                    row['classification_label'] = classification_label
                    row['classification_path'] = classification_path
                    row['top_concepts_json'] = top_concepts_json
                    row['classification_mode'] = str(self.classification_mode)
                    row['vector_hash'] = vector_hash
                    row['child1_embedding_string'] = child1_embedding_string
                    row['child2_embedding_string'] = child2_embedding_string
                    
                    # Write updated row
                    writer.writerow(row)
                    updated_count += 1
            
            # Save all caches for future use
            self.save_embeddings()
            
            logger.info(f"Updated {updated_count} rows in training dataset using {self.classification_mode} mode")
            logger.info(f"Classified {classified_count} rows with specific labels")
            logger.info(f"Updated dataset saved to {output_path}")
            
            # Return metrics
            metrics = {
                'status': 'completed',
                'total_rows': total_rows,
                'updated_rows': updated_count,
                'classified_rows': classified_count,
                'output_path': output_path,
                'delimiter_used': delimiter,
                'classification_mode': str(self.classification_mode)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating training dataset: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_similarity_report(self) -> Dict[str, Any]:
        """
        Generate a report of classification examples and similarity matching.
        The report includes examples of classifications for different composite fields
        and a breakdown of the distribution of classifications.
        
        Returns:
            Dictionary with report generation metrics
        """
        logger.info(f"Generating classification similarity report for {self.classification_mode} mode")
        
        # Define report output path
        report_path = os.path.join(self.output_dir, f"classification_report_{self.classification_mode}.json")
        
        try:
            # Load classified dataset
            classified_path = os.path.join(self.output_dir, "training_dataset_classified.csv")
            
            if not os.path.exists(classified_path):
                logger.error(f"Classified dataset not found at {classified_path}")
                return {'status': 'error', 'message': 'Classified dataset not found'}
            
            # Detect delimiter
            def detect_delimiter(file_path, num_lines=5):
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = ''.join([f.readline() for _ in range(num_lines)])
                
                # Count common delimiters
                delimiters = [',', '\t', ';', '|']
                counts = {d: sample.count(d) for d in delimiters}
                
                # Return the most common delimiter
                most_common = max(counts.items(), key=lambda x: x[1])[0]
                return most_common
            
            delimiter = detect_delimiter(classified_path)
            
            # Read classified dataset
            with open(classified_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
            
            # Only include rows classified with current mode
            mode_rows = [row for row in rows if row.get('classification_mode', '') == str(self.classification_mode)]
            if not mode_rows:
                logger.warning(f"No rows found with classification mode {self.classification_mode}")
                mode_rows = rows  # Fall back to all rows
            
            # Count classifications
            classification_counts = {}
            for row in mode_rows:
                label = row.get('classification_label', 'unknown')
                if label not in classification_counts:
                    classification_counts[label] = 0
                classification_counts[label] += 1
            
            # Sort classifications by count (descending)
            sorted_classifications = sorted(
                classification_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Select example rows for each classification (up to 5 per classification)
            examples = {}
            for label, _ in sorted_classifications:
                if label == 'unknown':
                    continue
                    
                # Find rows with this classification
                example_rows = [row for row in mode_rows if row.get('classification_label') == label]
                # Select up to 5 examples
                selected_examples = example_rows[:5]
                
                # Enhanced example details
                detailed_examples = []
                for row in selected_examples:
                    example = {
                        'composite': row.get('composite', ''),
                        'classification_path': row.get('classification_path', ''),
                        'personId': row.get('personId', '')
                    }
                    
                    # Try to parse top concepts JSON
                    try:
                        top_concepts_json = row.get('top_concepts_json', '[]')
                        top_concepts = json.loads(top_concepts_json)
                        example['top_concepts'] = top_concepts
                    except:
                        example['top_concepts'] = []
                    
                    detailed_examples.append(example)
                
                examples[label] = detailed_examples
            
            # Prepare report data
            report_data = {
                'classification_counts': dict(sorted_classifications),
                'total_records': len(mode_rows),
                'classified_records': len(mode_rows) - classification_counts.get('unknown', 0),
                'classification_examples': examples,
                'top_k_used': self.top_k,
                'classification_mode': str(self.classification_mode),
                'embedding_model': self.embedding_model
            }
            
            # Write report to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Classification report generated for {len(mode_rows)} records with {self.classification_mode} mode")
            logger.info(f"Report saved to {report_path}")
            
            # Return metrics
            metrics = {
                'status': 'completed',
                'total_records': len(mode_rows),
                'classified_records': len(mode_rows) - classification_counts.get('unknown', 0),
                'report_path': report_path,
                'classification_mode': str(self.classification_mode)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating similarity report: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'message': str(e)}
    
    def close(self):
        """
        Close connections and release resources.
        Ensures all resources are properly released in all scenarios.
        """
        try:
            # Check if client exists and is not None
            if hasattr(self, 'weaviate_client') and self.weaviate_client is not None:
                # Check if close method exists (defensive programming)
                if hasattr(self.weaviate_client, 'close'):
                    self.weaviate_client.close()
                    logger.info("Weaviate client connection closed")
                else:
                    logger.warning("Weaviate client has no close method")
                
                # Clear reference to avoid double-close attempts
                self.weaviate_client = None
            else:
                logger.debug("No Weaviate client to close")
                
            # Clean up OpenAI client if needed
            if hasattr(self, 'openai_client') and self.openai_client is not None:
                # No explicit close needed for OpenAI client, but clear reference
                self.openai_client = None
                
            logger.info("All connections and resources released")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

    def _create_classification_schema(self):
        """
        Create ConceptScheme collection in Weaviate.
        Forces recreation if in reset mode.
        """
        logger.info("Creating ConceptScheme collection in Weaviate")
        ef = self.config.get("weaviate_ef", 128)
        max_connections = self.config.get("weaviate_max_connections", 64)
        ef_construction = self.config.get("weaviate_ef_construction", 128)
        try:
            if self.reset_mode:
                try:
                    self.weaviate_client.collections.delete("ConceptScheme")
                    logger.info("Deleted existing ConceptScheme collection (reset mode)")
                except Exception as e:
                    logger.info(f"No existing ConceptScheme collection to delete: {str(e)}")
            else:
                recreate_collections = self.config.get("recreate_collections", False)
                if recreate_collections:
                    try:
                        self.weaviate_client.collections.delete("ConceptScheme")
                        logger.info("Deleted existing ConceptScheme collection (config setting)")
                    except Exception:
                        logger.info("No existing ConceptScheme collection found to delete")
                else:
                    try:
                        collection = self.weaviate_client.collections.get("ConceptScheme")
                        logger.info("ConceptScheme collection already exists")
                        return collection
                    except Exception:
                        pass
            collection = self.weaviate_client.collections.create(
                name="ConceptScheme",
                description="Collection for knowledge classification concepts with their embeddings",
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    ef=ef,
                    max_connections=max_connections,
                    ef_construction=ef_construction,
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="concept_uri", data_type=DataType.TEXT),
                    Property(name="pref_label", data_type=DataType.TEXT),
                    Property(name="alt_labels", data_type=DataType.TEXT_ARRAY),
                    Property(name="definition", data_type=DataType.TEXT),
                    Property(name="scope_note", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT_ARRAY),
                    Property(name="embedding_string", data_type=DataType.TEXT)
                ]
            )
            logger.info(f"Created ConceptScheme collection with HNSW vector index")
            return collection
        except Exception as e:
            logger.error(f"Error creating ConceptScheme schema: {str(e)}")
            raise

    def build_hierarchical_embedding_string(self, taxonomy_data, uri, labels, definitions, scope_notes, alt_labels, parent_lookup):
        def build_embedding_string(u):
            label = labels.get(u, '')
            embedding_text = f"{label}"
            definition = definitions.get(u)
            if definition:
                embedding_text += f": {definition}"
            scope_note = scope_notes.get(u)
            if scope_note:
                embedding_text += f" [{scope_note}]"
            alts = alt_labels.get(u, [])
            if alts:
                embedding_text += f". Also known as: {', '.join([a for a in alts if a])}"
            return embedding_text
        parent = parent_lookup.get(uri)
        if parent:
            parent_str = self.build_hierarchical_embedding_string(taxonomy_data, parent, labels, definitions, scope_notes, alt_labels, parent_lookup)
            return f"{parent_str} | {build_embedding_string(uri)}"
        else:
            return build_embedding_string(uri)

def run_diagnostic_mode(config_path: str, person_id1: str, person_id2: str, mode: str = None, 
                      top_k: int = None, process_ids: List[str] = None, vector_mode: str = None):
    """
    Run diagnostic mode to compare two person IDs.
    
    Args:
        config_path: Path to the configuration file
        person_id1: First person ID
        person_id2: Second person ID
        mode: Classification mode to use (average or filter)
        top_k: Optional number of top concepts to use (defaults to config value)
        process_ids: Optional list of specific person IDs to process first
        vector_mode: Vector similarity mode to use (average or top)
    """
    processor = None
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set top_k in config if provided
        if top_k is not None:
            config['top_k'] = top_k
        
        # Set classification mode if provided
        if mode is not None:
            config['classification_mode'] = mode
        
        # Initialize processor
        processor = ClassificationEmbedding(config)
        
        # Load cached embeddings and classifications
        processor.load_embeddings()
        
        # Process specific IDs if provided
        if process_ids:
            for pid in process_ids:
                result = processor.process_specific_person_id(pid)
                if result['status'] != 'success':
                    logger.warning(f"Failed to process person ID {pid}: {result['message']}")
        
        # Compare in both modes if not specified
        if mode is None:
            # Compare in average mode
            comparison_avg = processor.compare_person_ids(person_id1, person_id2, ClassificationMode.AVERAGE, vector_mode)
            
            # Compare in filter mode
            comparison_filter = processor.compare_person_ids(person_id1, person_id2, ClassificationMode.FILTER, vector_mode)
            
            # Print results for both modes
            if comparison_avg['status'] == 'success':
                print("\n=== Person ID Comparison (AVERAGE MODE) ===")
                print(f"Similarity: {comparison_avg['vector_similarity']:.4f}")
                print("\nPerson ID 1:")
                print(f"ID: {comparison_avg['person_id1']['id']}")
                print(f"Composite: {comparison_avg['person_id1']['composite']}")
                print(f"Primary Concept: {comparison_avg['person_id1']['primary_concept_label']}")
                print(f"Concept Path: {comparison_avg['person_id1']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison_avg['person_id1']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
                
                print("\nPerson ID 2:")
                print(f"ID: {comparison_avg['person_id2']['id']}")
                print(f"Composite: {comparison_avg['person_id2']['composite']}")
                print(f"Primary Concept: {comparison_avg['person_id2']['primary_concept_label']}")
                print(f"Concept Path: {comparison_avg['person_id2']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison_avg['person_id2']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
            else:
                print(f"Error (AVERAGE MODE): {comparison_avg['message']}")
            
            if comparison_filter['status'] == 'success':
                print("\n=== Person ID Comparison (FILTER MODE) ===")
                print(f"Similarity: {comparison_filter['vector_similarity']:.4f}")
                print("\nPerson ID 1:")
                print(f"ID: {comparison_filter['person_id1']['id']}")
                print(f"Composite: {comparison_filter['person_id1']['composite']}")
                print(f"Primary Concept: {comparison_filter['person_id1']['primary_concept_label']}")
                print(f"Concept Path: {comparison_filter['person_id1']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison_filter['person_id1']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
                
                print("\nPerson ID 2:")
                print(f"ID: {comparison_filter['person_id2']['id']}")
                print(f"Composite: {comparison_filter['person_id2']['composite']}")
                print(f"Primary Concept: {comparison_filter['person_id2']['primary_concept_label']}")
                print(f"Concept Path: {comparison_filter['person_id2']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison_filter['person_id2']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
            else:
                print(f"Error (FILTER MODE): {comparison_filter['message']}")
            
            # Return both comparison results
            return {
                'average_mode': comparison_avg,
                'filter_mode': comparison_filter
            }
        else:
            # Use specified mode
            classification_mode = ClassificationMode(mode)
            comparison = processor.compare_person_ids(person_id1, person_id2, classification_mode, vector_mode)
            
            if comparison['status'] == 'success':
                print(f"\n=== Person ID Comparison ({classification_mode.upper()} MODE) ===")
                print(f"Similarity: {comparison['vector_similarity']:.4f}")
                print("\nPerson ID 1:")
                print(f"ID: {comparison['person_id1']['id']}")
                print(f"Composite: {comparison['person_id1']['composite']}")
                print(f"Primary Concept: {comparison['person_id1']['primary_concept_label']}")
                print(f"Concept Path: {comparison['person_id1']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison['person_id1']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
                
                print("\nPerson ID 2:")
                print(f"ID: {comparison['person_id2']['id']}")
                print(f"Composite: {comparison['person_id2']['composite']}")
                print(f"Primary Concept: {comparison['person_id2']['primary_concept_label']}")
                print(f"Concept Path: {comparison['person_id2']['primary_concept_path']}")
                print("Top Concepts:")
                for concept in comparison['person_id2']['top_concepts']:
                    print(f"  - {concept['pref_label']} (Similarity: {concept['similarity']:.4f})")
            else:
                print(f"Error: {comparison['message']}")
            
            return comparison
        
    except Exception as e:
        logger.error(f"Error in diagnostic mode: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'status': 'error', 'message': str(e)}
    finally:
        # Ensure all connections are released
        if processor is not None:
            processor.close()

def main(config_path: str = 'config.yml', reset_mode: bool = False, top_k: int = None, 
       mode: str = None, vector_mode: str = None):
    """
    Main function for the classification embedding script.
    Implements robust execution flow with proper resource management.
    
    Args:
        config_path: Path to the configuration file
        reset_mode: Flag to force reset of collections and caches
        top_k: Optional override for the number of top concepts to consider
        mode: Classification mode to use (average or filter)
        vector_mode: Vector similarity mode to use (average or top)
    """
    processor = None
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set top_k in config if provided
        if top_k is not None:
            config['top_k'] = top_k
        
        # Set classification mode if provided
        if mode is not None:
            config['classification_mode'] = mode
        
        # Initialize classification embedding processor with reset mode
        processor = ClassificationEmbedding(config, reset_mode=reset_mode)
        
        # Data processing stage
        # ---------------------
        
        # Load classification scheme from SKOS taxonomy
        classification_scheme = processor.load_classification_scheme()
        
        # Check if embeddings already exist - will be skipped if in reset mode
        embeddings = processor.load_embeddings()
        
        if not embeddings:
            # Generate embeddings
            embeddings = processor.generate_embeddings(classification_scheme)
            
            # Save embeddings
            processor.save_embeddings()
        
        # Index embeddings in Weaviate - will handle reset mode appropriately
        indexing_metrics = processor.index_embeddings(classification_scheme)
        
        # Update training dataset with classification labels
        update_metrics = processor.update_training_dataset()
        
        # Generate similarity report
        similarity_report_metrics = processor.generate_similarity_report()
        
        # Return overall metrics
        metrics = {
            'status': 'completed',
            'embedding_count': len(embeddings),
            'indexing_metrics': indexing_metrics,
            'update_metrics': update_metrics,
            'similarity_report_metrics': similarity_report_metrics,
            'reset_mode': reset_mode,
            'top_k': processor.top_k,
            'classification_mode': str(processor.classification_mode)
        }
        
        logger.info(f"Classification embedding script completed successfully with {processor.classification_mode} mode" + 
                   (" in RESET mode" if reset_mode else ""))
        logger.info(f"Overall metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in classification embedding script: {str(e)}")
        return {'status': 'error', 'message': str(e)}
    finally:
        # Ensure Weaviate connection is closed in all cases
        if processor is not None:
            try:
                processor.close()
                logger.info("Weaviate connection closed properly")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {str(e)}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Knowledge Classification Embedding Script')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--reset', action='store_true', help='Delete collections and caches to force regeneration')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--diagnostic', nargs=2, metavar=('PERSON_ID1', 'PERSON_ID2'), 
                        help='Run in diagnostic mode to compare two personId values')
    parser.add_argument('--top_k', type=int, help='Number of top concepts to consider (default: 5)')
    parser.add_argument('--mode', choices=['average', 'filter'], 
                        help='Classification mode: "average" (vector averaging) or "filter" (concept filtering)')
    parser.add_argument('--vector-mode', choices=['average', 'top'], default='average',
                        help='Vector similarity mode: "average" (default) uses averaged top concept vectors; "top" uses only the top concept vector for each entity')
    parser.add_argument('--process_ids', help='Comma-separated list of person IDs to process in diagnostic mode')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Log reset mode if enabled
    if args.reset:
        logger.info("RESET MODE ENABLED: Will delete collections and caches for fresh generation")
    
    # Parse process_ids if provided
    process_ids = None
    if args.process_ids:
        process_ids = args.process_ids.split(',')
    
    # Run in appropriate mode
    try:
        if args.diagnostic:
            # Run in diagnostic mode
            run_diagnostic_mode(args.config, args.diagnostic[0], args.diagnostic[1], 
                              args.mode, args.top_k, process_ids, args.vector_mode)
        else:
            # Run in normal mode
            main(args.config, reset_mode=args.reset, top_k=args.top_k, mode=args.mode, vector_mode=args.vector_mode)
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {str(e)}")
    finally:
        # Ensure all connections are closed
        logger.info("Script execution completed")

import json

def extract_label(label):
    if isinstance(label, dict):
        return label.get('@value')
    elif isinstance(label, list):
        for entry in label:
            if isinstance(entry, dict) and entry.get('@language') == 'en':
                return entry.get('@value')
        if label and isinstance(label[0], dict):
            return label[0].get('@value')
        elif label:
            return label[0]
    return label

# --- NEW: Recursively collect all concepts and parent relationships ---
def collect_concepts_and_parents(node, parent_uri=None, all_concepts=None, parent_lookup=None):
    if all_concepts is None:
        all_concepts = {}
    if parent_lookup is None:
        parent_lookup = {}
    if isinstance(node, dict):
        if node.get('@id') and node.get('@type') == 'skos:Concept':
            uri = node['@id']
            if uri not in all_concepts:
                all_concepts[uri] = node
            if parent_uri:
                if uri not in parent_lookup:
                    parent_lookup[uri] = parent_uri
        for k, v in node.items():
            if k == 'skos:narrower':
                if isinstance(v, list):
                    for child in v:
                        collect_concepts_and_parents(child, node.get('@id'), all_concepts, parent_lookup)
                else:
                    collect_concepts_and_parents(v, node.get('@id'), all_concepts, parent_lookup)
            elif isinstance(v, dict) or isinstance(v, list):
                collect_concepts_and_parents(v, parent_uri, all_concepts, parent_lookup)
    elif isinstance(node, list):
        for item in node:
            collect_concepts_and_parents(item, parent_uri, all_concepts, parent_lookup)
    return all_concepts, parent_lookup

# --- Replace the old concept table logic with the robust recursive version ---
# Removed global taxonomy.json reads
