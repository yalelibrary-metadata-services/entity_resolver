#!/usr/bin/env python3
"""
pickle2json.py - Convert Pickle files to JSON format

Usage:
    python pickle2json.py input.pkl [output.json] [--indent SPACES] [--sort-keys] [--verbose]
"""

import argparse
import json
import pickle
import os
import sys
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import gc


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling objects not serializable by default."""
    
    def __init__(self, *args, max_depth=100, current_depth=0, **kwargs):
        self.max_depth = max_depth
        self.current_depth = current_depth
        super().__init__(*args, **kwargs)
    
    def default(self, obj):
        # Check recursion depth
        if self.current_depth >= self.max_depth:
            return {"_type": "max_depth_reached", "value": str(type(obj))}
        
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return {"_type": "datetime", "value": obj.isoformat()}
        
        # Handle timedelta objects
        if isinstance(obj, timedelta):
            return {"_type": "timedelta", "value": str(obj)}
        
        # Handle sets
        if isinstance(obj, set):
            return {"_type": "set", "value": list(obj)}
        
        # Handle bytes
        if isinstance(obj, bytes):
            try:
                return {"_type": "bytes", "value": obj.decode('utf-8', errors='replace')}
            except:
                return {"_type": "bytes", "value": str(obj)}
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return {"_type": "ndarray", "value": obj.tolist()}
        
        # Handle numpy data types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle other objects
        try:
            # Try to convert to dictionary if object has __dict__ attribute
            if hasattr(obj, "__dict__"):
                obj_dict = obj.__dict__.copy()
                obj_dict["_type"] = obj.__class__.__name__
                return obj_dict
            
            # For other objects, convert to string
            return {"_type": obj.__class__.__name__, "value": str(obj)}
        except:
            return {"_type": "unknown", "value": str(type(obj))}


def pickle_to_json(pickle_path, json_path=None, indent=2, sort_keys=False, verbose=False, compact=False, max_depth=100):
    """Convert a Pickle file to JSON format."""
    # Default output path if not specified
    if json_path is None:
        json_path = os.path.splitext(pickle_path)[0] + '.json'
    
    try:
        # Load data from Pickle file
        if verbose:
            print(f"Loading data from '{pickle_path}'...")
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if verbose:
            print(f"Data loaded successfully. Converting to JSON...")
        
        # Use compact format if requested
        use_indent = None if compact else indent
        
        # Convert data to JSON and write to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=use_indent, sort_keys=sort_keys)
        
        # Clean up memory
        del data
        gc.collect()
        
        # Get output file size
        output_size = os.path.getsize(json_path)
        
        if verbose:
            print(f"Data converted and written to '{json_path}'")
            print(f"Output file size: {output_size / (1024*1024):.2f} MB")
        else:
            print(f"Successfully converted '{pickle_path}' to '{json_path}'")
        
        return json_path
    
    except (pickle.PickleError, json.JSONDecodeError) as e:
        print(f"Error: Failed to process pickle data: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Convert a Pickle file to JSON format.')
    parser.add_argument('input', help='Path to the input Pickle file')
    parser.add_argument('output', nargs='?', help='Path to the output JSON file (optional)')
    parser.add_argument('--indent', type=int, default=2, help='Number of spaces for indentation (default: 2)')
    parser.add_argument('--sort-keys', action='store_true', help='Sort dictionary keys')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during conversion')
    parser.add_argument('--compact', action='store_true', help='Use compact JSON format (no indentation)')
    parser.add_argument('--max-depth', type=int, default=100, help='Maximum recursion depth for nested objects')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Convert Pickle to JSON
    pickle_to_json(
        args.input, 
        args.output, 
        args.indent, 
        args.sort_keys, 
        args.verbose, 
        args.compact, 
        args.max_depth
    )


if __name__ == '__main__':
    main()