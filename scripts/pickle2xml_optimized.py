#!/usr/bin/env python3
"""
pickle2xml_optimized.py - Memory-efficient conversion of Pickle files to XML format

Usage:
    python pickle2xml_optimized.py input.pkl [output.xml] [--chunk-size SIZE] [--verbose]
"""

import argparse
import pickle
import os
import sys
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import gc
import xml.sax.saxutils as saxutils
from io import StringIO


class StreamingXMLWriter:
    """Memory-efficient XML writer that streams directly to file."""
    
    def __init__(self, output_file, max_depth=100, chunk_size=1000):
        self.output_file = output_file
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.current_depth = 0
        self.writer = saxutils.XMLGenerator(output_file, encoding='utf-8')
        self.writer.startDocument()
    
    def write_element(self, obj, element_name="item"):
        """Write Python object as XML element directly to file."""
        self._write_element_recursive(obj, element_name, 0)
    
    def _write_element_recursive(self, obj, element_name, depth):
        """Recursively write elements with depth control."""
        if depth >= self.max_depth:
            attrs = {'type': 'max_depth_reached'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(type(obj)))
            self.writer.endElement(element_name)
            return
        
        # Handle None
        if obj is None:
            attrs = {'type': 'null'}
            self.writer.startElement(element_name, attrs)
            self.writer.endElement(element_name)
            return
        
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            attrs = {'type': 'datetime'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(obj.isoformat())
            self.writer.endElement(element_name)
            return
        
        # Handle timedelta objects
        if isinstance(obj, timedelta):
            attrs = {'type': 'timedelta'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(obj))
            self.writer.endElement(element_name)
            return
        
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            attrs = {'type': type(obj).__name__}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(obj))
            self.writer.endElement(element_name)
            return
        
        # Handle lists and tuples - CHUNKED PROCESSING
        if isinstance(obj, (list, tuple)):
            attrs = {'type': type(obj).__name__, 'length': str(len(obj))}
            self.writer.startElement(element_name, attrs)
            
            # Process in chunks to avoid memory buildup
            for i in range(0, len(obj), self.chunk_size):
                chunk = obj[i:i + self.chunk_size]
                for j, item in enumerate(chunk):
                    self._write_element_recursive(item, f"item_{i+j}", depth + 1)
                # Force garbage collection after each chunk
                del chunk
                gc.collect()
            
            self.writer.endElement(element_name)
            return
        
        # Handle sets
        if isinstance(obj, set):
            attrs = {'type': 'set', 'length': str(len(obj))}
            self.writer.startElement(element_name, attrs)
            
            # Convert to list and process in chunks
            obj_list = list(obj)
            for i in range(0, len(obj_list), self.chunk_size):
                chunk = obj_list[i:i + self.chunk_size]
                for j, item in enumerate(chunk):
                    self._write_element_recursive(item, f"item_{i+j}", depth + 1)
                del chunk
                gc.collect()
            
            del obj_list
            self.writer.endElement(element_name)
            return
        
        # Handle dictionaries - CHUNKED PROCESSING
        if isinstance(obj, dict):
            attrs = {'type': 'dict', 'length': str(len(obj))}
            self.writer.startElement(element_name, attrs)
            
            # Process dictionary in chunks
            items = list(obj.items())
            for i in range(0, len(items), self.chunk_size):
                chunk = items[i:i + self.chunk_size]
                for key, value in chunk:
                    key_name = self._sanitize_key(str(key))
                    key_attrs = {'key': str(key)}
                    # Write key as attribute and process value
                    self._write_element_with_attrs(value, key_name, key_attrs, depth + 1)
                del chunk
                gc.collect()
            
            del items
            self.writer.endElement(element_name)
            return
        
        # Handle bytes
        if isinstance(obj, bytes):
            attrs = {'type': 'bytes'}
            self.writer.startElement(element_name, attrs)
            try:
                self.writer.characters(obj.decode('utf-8', errors='replace'))
            except:
                self.writer.characters(str(obj))
            self.writer.endElement(element_name)
            return
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            attrs = {'type': 'ndarray', 'shape': str(obj.shape), 'dtype': str(obj.dtype)}
            self.writer.startElement(element_name, attrs)
            # Convert to list and process recursively
            self._write_element_recursive(obj.tolist(), "array_data", depth + 1)
            self.writer.endElement(element_name)
            return
        
        # Handle numpy data types
        if isinstance(obj, np.integer):
            attrs = {'type': 'numpy_int'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(int(obj)))
            self.writer.endElement(element_name)
            return
        if isinstance(obj, np.floating):
            attrs = {'type': 'numpy_float'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(float(obj)))
            self.writer.endElement(element_name)
            return
        if isinstance(obj, np.bool_):
            attrs = {'type': 'numpy_bool'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(bool(obj)))
            self.writer.endElement(element_name)
            return
        
        # Handle other objects
        try:
            if hasattr(obj, "__dict__"):
                attrs = {'type': 'object', 'class': obj.__class__.__name__}
                self.writer.startElement(element_name, attrs)
                
                # Process object attributes in chunks
                items = list(obj.__dict__.items())
                for i in range(0, len(items), self.chunk_size):
                    chunk = items[i:i + self.chunk_size]
                    for key, value in chunk:
                        key_name = self._sanitize_key(str(key))
                        key_attrs = {'key': str(key)}
                        self._write_element_with_attrs(value, key_name, key_attrs, depth + 1)
                    del chunk
                    gc.collect()
                
                del items
                self.writer.endElement(element_name)
                return
            
            # For other objects, convert to string
            attrs = {'type': obj.__class__.__name__}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(obj))
            self.writer.endElement(element_name)
            return
        except:
            attrs = {'type': 'unknown'}
            self.writer.startElement(element_name, attrs)
            self.writer.characters(str(type(obj)))
            self.writer.endElement(element_name)
            return
    
    def _write_element_with_attrs(self, obj, element_name, attrs, depth):
        """Write element with additional attributes."""
        # This is a simplified version - in a full implementation,
        # we'd need to merge attributes with those generated by the recursive call
        self._write_element_recursive(obj, element_name, depth)
    
    def _sanitize_key(self, key):
        """Sanitize dictionary key to be valid XML element name."""
        sanitized = ""
        for char in key:
            if char.isalnum() or char == "_":
                sanitized += char
            else:
                sanitized += "_"
        
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "key_" + sanitized
        
        return sanitized
    
    def close(self):
        """Close the XML document."""
        self.writer.endDocument()


def pickle_to_xml_optimized(pickle_path, xml_path=None, chunk_size=1000, verbose=False, max_depth=100):
    """Convert a Pickle file to XML format with memory optimization."""
    if xml_path is None:
        xml_path = os.path.splitext(pickle_path)[0] + '_optimized.xml'
    
    try:
        if verbose:
            print(f"Loading data from '{pickle_path}'...")
        
        # Load data from Pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if verbose:
            print(f"Data loaded. Type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"Data length: {len(data)}")
            print(f"Converting to XML with chunk size {chunk_size}...")
        
        # Write XML using streaming approach
        with open(xml_path, 'w', encoding='utf-8') as f:
            writer = StreamingXMLWriter(f, max_depth=max_depth, chunk_size=chunk_size)
            writer.write_element(data, "root")
            writer.close()
        
        # Clean up memory
        del data
        gc.collect()
        
        # Get output file size
        output_size = os.path.getsize(xml_path)
        
        if verbose:
            print(f"Data converted and written to '{xml_path}'")
            print(f"Output file size: {output_size / (1024*1024):.2f} MB")
        else:
            print(f"Successfully converted '{pickle_path}' to '{xml_path}'")
        
        return xml_path
    
    except pickle.PickleError as e:
        print(f"Error: Failed to process pickle data: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Convert a Pickle file to XML format (memory optimized).')
    parser.add_argument('input', help='Path to the input Pickle file')
    parser.add_argument('output', nargs='?', help='Path to the output XML file (optional)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Number of items to process in each chunk')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during conversion')
    parser.add_argument('--max-depth', type=int, default=100, help='Maximum recursion depth for nested objects')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Convert Pickle to XML
    pickle_to_xml_optimized(
        args.input, 
        args.output, 
        args.chunk_size, 
        args.verbose, 
        args.max_depth
    )


if __name__ == '__main__':
    main()