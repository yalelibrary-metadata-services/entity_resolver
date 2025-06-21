"""
Preprocessing Module for Entity Resolution

This module handles data preprocessing tasks for entity resolution, including:
- Loading and parsing input data files
- Deduplicating field values
- Creating hash-based data structures for efficient storage
- Maintaining mappings between unique strings and entity identifiers
"""

import os
import csv
import logging
import pickle
import hashlib
import time
from typing import Dict, List, Tuple, Any, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

def hash_string(string_value: str) -> str:
    """
    Create a deterministic hash for a string value.
    
    Args:
        string_value: String value to hash
        
    Returns:
        MD5 hash of the string
    """
    if not string_value:
        return "NULL"
        
    # Create deterministic hash
    return hashlib.md5(string_value.encode('utf-8')).hexdigest()

def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a single input file and build data structures.
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        Dictionary with processing metrics and data structures
    """
    start_time = time.time()
    
    # Initialize local data structures
    local_output_data = {}
    local_string_counts = {}
    local_field_hash_mapping = {}
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, na_values='', keep_default_na=False)
        
        # Make sure all expected fields are present
        required_fields = ['composite', 'marcKey', 'person', 'roles', 'title', 'provision', 'subjects', 'personId']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.warning(f"File {file_path} is missing fields: {missing_fields}")
            # Fill missing fields with empty strings
            for field in missing_fields:
                df[field] = ''
        
        # Process each row
        total_rows = 0
        for _, row in df.iterrows():
            total_rows += 1
            
            # Get personId
            person_id = str(row['personId'])
            if not person_id:
                continue
                
            # Initialize field hashes for this record
            local_output_data[person_id] = {}
            
            # Process each field
            for field in required_fields:
                if field == 'personId':
                    continue
                    
                # Get field value
                value = str(row[field]) if pd.notna(row[field]) else ''
                
                # Hash the value
                hash_val = hash_string(value)
                
                # Update field hashes
                local_output_data[person_id][field] = hash_val
                
                # Update string counts
                if hash_val != "NULL":
                    local_string_counts[hash_val] = local_string_counts.get(hash_val, 0) + 1
                    
                    # Update field/hash mapping
                    if hash_val not in local_field_hash_mapping:
                        local_field_hash_mapping[hash_val] = {}
                    local_field_hash_mapping[hash_val][field] = local_field_hash_mapping[hash_val].get(field, 0) + 1
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Processed {file_path} with {total_rows} rows in {elapsed_time:.2f} seconds")
        
        return {
            'file_path': file_path,
            'rows_processed': total_rows,
            'elapsed_time': elapsed_time,
            'output_data': local_output_data,
            'string_counts': local_string_counts,
            'field_hash_mapping': local_field_hash_mapping
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'rows_processed': 0,
            'error': str(e),
            'output_data': {},
            'string_counts': {},
            'field_hash_mapping': {}
        }

def create_string_dict(string_counts: Dict[str, int], field_hash_mapping: Dict[str, Dict[str, int]], input_dir: str) -> Dict[str, str]:
    """
    Create dictionary of hash values to original string values.
    
    Args:
        string_counts: Dictionary tracking string occurrence counts
        field_hash_mapping: Dictionary mapping field/hash to usage counts
        input_dir: Directory containing input files
        
    Returns:
        Dictionary mapping hash to original string value
    """
    logger.info("Building string dictionary from hash values")
    string_dict = {}
    
    # Initialize hash-to-string mapping with reverse lookup tables
    hash_to_string_map = {}
    
    # Get list of input files
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    logger.info(f"Scanning {len(csv_files)} CSV files to extract original string values")
    
    # Process each CSV file to extract original string values
    file_count = 0
    for csv_file in csv_files:
        try:
            file_count += 1
            if file_count % 10 == 0:
                logger.info(f"Processed {file_count} files out of {len(csv_files)}")
                
            # Load the CSV file with proper encoding
            df = pd.read_csv(csv_file, na_values='', keep_default_na=False)
            
            # Make sure all expected fields are present
            required_fields = ['composite', 'marcKey', 'person', 'roles', 'title', 'provision', 'subjects', 'personId']
            for field in required_fields:
                if field == 'personId':  # Skip personId - not a string we want to record
                    continue
                    
                if field not in df.columns:
                    continue
                    
                # Process each value in this field
                for value in df[field].dropna().astype(str).unique():
                    if not value:  # Skip empty strings
                        continue
                        
                    # Calculate hash
                    hash_val = hash_string(value)
                    
                    # Record the original string value
                    if hash_val != "NULL" and hash_val in string_counts:
                        hash_to_string_map[hash_val] = value
                        
        except Exception as e:
            logger.warning(f"Error extracting strings from {csv_file}: {str(e)}")
    
    # Create the final string dictionary
    for hash_val in string_counts.keys():
        # Use the extracted string if available, otherwise use empty string
        string_dict[hash_val] = hash_to_string_map.get(hash_val, f"Unknown string with hash {hash_val}")
    
    # Log statistics
    filled_count = sum(1 for val in string_dict.values() if val and not val.startswith("Unknown string"))
    logger.info(f"Built string dictionary with {len(string_dict)} entries, {filled_count} with extracted values")
    
    return string_dict

def process_data(config: Dict[str, Any], input_dir: str, checkpoint_dir: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Any]]:
    """
    Process input data files and create data structures for entity resolution.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing input files
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        Tuple of (hash_lookup, string_dict, metrics)
    """
    logger.info(f"Starting data preprocessing from {input_dir}")
    start_time = time.time()
    
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize data structures
    output_data = {}  # personId -> field -> hash
    string_counts = {}  # hash -> count
    field_hash_mapping = {}  # hash -> field -> count
    
    # Get list of input files
    input_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            input_files.append(os.path.join(input_dir, file))
    
    if not input_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {}, {}, {'status': 'no_files_found'}
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Process files in parallel
    total_rows = 0
    file_metrics = []
    batch_size = config.get("preprocessing_batch_size", 10)
    max_workers = config.get("preprocessing_workers", 4)
    
    # Process in batches to avoid memory issues
    for i in range(0, len(input_files), batch_size):
        batch_files = input_files[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(input_files)-1)//batch_size + 1} with {len(batch_files)} files")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_file, file_path): file_path
                for file_path in batch_files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(batch_files), desc="Processing files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    
                    # Merge batch results into main data structures
                    if 'output_data' in result:
                        for person_id, fields in result['output_data'].items():
                            output_data[person_id] = fields
                            
                    if 'string_counts' in result:
                        for hash_val, count in result['string_counts'].items():
                            string_counts[hash_val] = string_counts.get(hash_val, 0) + count
                            
                    if 'field_hash_mapping' in result:
                        for hash_val, fields in result['field_hash_mapping'].items():
                            if hash_val not in field_hash_mapping:
                                field_hash_mapping[hash_val] = {}
                            for field, count in fields.items():
                                field_hash_mapping[hash_val][field] = field_hash_mapping[hash_val].get(field, 0) + count
                    
                    # Track metrics
                    total_rows += result.get('rows_processed', 0)
                    file_metrics.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing results for {file_path}: {str(e)}")
        
        # Save checkpoint after each batch
        save_checkpoint(output_data, string_counts, field_hash_mapping, checkpoint_dir, i // batch_size)
    
    # Create string dictionary
    string_dict = create_string_dict(string_counts, field_hash_mapping, input_dir)
    
    # Save final data structures
    hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
    string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
    string_counts_path = os.path.join(checkpoint_dir, 'string_counts.pkl')
    field_hash_mapping_path = os.path.join(checkpoint_dir, 'field_hash_mapping.pkl')
    
    with open(hash_lookup_path, 'wb') as f:
        pickle.dump(output_data, f)
        
    with open(string_dict_path, 'wb') as f:
        pickle.dump(string_dict, f)
        
    with open(string_counts_path, 'wb') as f:
        pickle.dump(string_counts, f)
        
    with open(field_hash_mapping_path, 'wb') as f:
        pickle.dump(field_hash_mapping, f)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {total_rows} total rows")
    logger.info(f"Found {len(output_data)} unique entities")
    logger.info(f"Found {len(string_counts)} unique strings")
    
    # Return final data structures and metrics
    metrics = {
        'status': 'completed',
        'elapsed_time': elapsed_time,
        'total_files': len(input_files),
        'total_rows': total_rows,
        'entity_count': len(output_data),
        'unique_strings': len(string_counts),
        'file_metrics': file_metrics
    }
    
    return output_data, string_dict, metrics

def save_checkpoint(output_data: Dict[str, Dict[str, str]], string_counts: Dict[str, int],
                  field_hash_mapping: Dict[str, Dict[str, int]], checkpoint_dir: str, batch_num: int) -> None:
    """
    Save checkpoint data during processing.
    
    Args:
        output_data: Dictionary of personId to field hashes
        string_counts: Dictionary of string counts
        field_hash_mapping: Dictionary mapping field/hash to usage counts
        checkpoint_dir: Directory for saving checkpoints
        batch_num: Batch number for checkpoint
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'preprocessing_checkpoint_{batch_num}.pkl')
    
    checkpoint_data = {
        'output_data': output_data,
        'string_counts': string_counts,
        'field_hash_mapping': field_hash_mapping,
        'timestamp': time.time()
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
        
    logger.debug(f"Saved preprocessing checkpoint to {checkpoint_path}")

def load_hash_lookup(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load hash lookup data from checkpoint.
    
    Args:
        path: Path to the hash lookup pickle file
        
    Returns:
        Dictionary mapping personId to field hashes
    """
    try:
        with open(path, 'rb') as f:
            hash_lookup = pickle.load(f)
        
        logger.info(f"Loaded hash lookup with {len(hash_lookup)} entities from {path}")
        return hash_lookup
        
    except Exception as e:
        logger.error(f"Error loading hash lookup from {path}: {str(e)}")
        return {}

def load_string_dict(path: str) -> Dict[str, str]:
    """
    Load string dictionary from checkpoint.
    
    Args:
        path: Path to the string dictionary pickle file
        
    Returns:
        Dictionary mapping hash to string value
    """
    try:
        with open(path, 'rb') as f:
            string_dict = pickle.load(f)
        
        logger.info(f"Loaded string dictionary with {len(string_dict)} entries from {path}")
        return string_dict
        
    except Exception as e:
        logger.error(f"Error loading string dictionary from {path}: {str(e)}")
        return {}

def load_string_counts(path: str) -> Dict[str, int]:
    """
    Load string counts from checkpoint.
    
    Args:
        path: Path to the string counts pickle file
        
    Returns:
        Dictionary mapping hash to count
    """
    try:
        with open(path, 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded string counts with {len(string_counts)} entries from {path}")
        return string_counts
        
    except Exception as e:
        logger.error(f"Error loading string counts from {path}: {str(e)}")
        return {}

def main(config_path: str = 'config.yml'):
    """
    Main function for running preprocessing.
    
    Args:
        config_path: Path to the configuration file
    """
    # This function would typically be called from the orchestrator
    # Implemented here for standalone module testing
    pass

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Preprocess data for entity resolution')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--input-dir', help='Directory containing input files')
    parser.add_argument('--checkpoint-dir', help='Directory for saving checkpoints')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.input_dir:
        config['input_dir'] = args.input_dir
    
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run preprocessing
    main(args.config)
