#!/usr/bin/env python3
"""
pickle2xml.py - Convert Pickle files to XML format

Usage:
    python pickle2xml.py input.pkl [output.xml] [--indent SPACES] [--sort-keys] [--verbose]
"""

import argparse
import pickle
import os
import sys
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import gc
import xml.etree.ElementTree as ET
from xml.dom import minidom


class CustomXMLEncoder:
    """Custom XML encoder for handling objects not serializable by default."""
    
    def __init__(self, max_depth=100, current_depth=0):
        self.max_depth = max_depth
        self.current_depth = current_depth
    
    def encode(self, obj, element_name="item"):
        """Convert Python object to XML element."""
        # Check recursion depth
        if self.current_depth >= self.max_depth:
            elem = ET.Element(element_name)
            elem.set("type", "max_depth_reached")
            elem.text = str(type(obj))
            return elem
        
        # Handle None
        if obj is None:
            elem = ET.Element(element_name)
            elem.set("type", "null")
            return elem
        
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            elem = ET.Element(element_name)
            elem.set("type", "datetime")
            elem.text = obj.isoformat()
            return elem
        
        # Handle timedelta objects
        if isinstance(obj, timedelta):
            elem = ET.Element(element_name)
            elem.set("type", "timedelta")
            elem.text = str(obj)
            return elem
        
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            elem = ET.Element(element_name)
            elem.set("type", type(obj).__name__)
            elem.text = str(obj)
            return elem
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            elem = ET.Element(element_name)
            elem.set("type", type(obj).__name__)
            encoder = CustomXMLEncoder(self.max_depth, self.current_depth + 1)
            for i, item in enumerate(obj):
                child = encoder.encode(item, f"item_{i}")
                elem.append(child)
            return elem
        
        # Handle sets
        if isinstance(obj, set):
            elem = ET.Element(element_name)
            elem.set("type", "set")
            encoder = CustomXMLEncoder(self.max_depth, self.current_depth + 1)
            for i, item in enumerate(obj):
                child = encoder.encode(item, f"item_{i}")
                elem.append(child)
            return elem
        
        # Handle dictionaries
        if isinstance(obj, dict):
            elem = ET.Element(element_name)
            elem.set("type", "dict")
            encoder = CustomXMLEncoder(self.max_depth, self.current_depth + 1)
            for key, value in obj.items():
                # Sanitize key for XML element name
                key_name = self._sanitize_key(str(key))
                child = encoder.encode(value, key_name)
                child.set("key", str(key))
                elem.append(child)
            return elem
        
        # Handle bytes
        if isinstance(obj, bytes):
            elem = ET.Element(element_name)
            elem.set("type", "bytes")
            try:
                elem.text = obj.decode('utf-8', errors='replace')
            except:
                elem.text = str(obj)
            return elem
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            elem = ET.Element(element_name)
            elem.set("type", "ndarray")
            elem.set("shape", str(obj.shape))
            elem.set("dtype", str(obj.dtype))
            encoder = CustomXMLEncoder(self.max_depth, self.current_depth + 1)
            list_child = encoder.encode(obj.tolist(), "array_data")
            elem.append(list_child)
            return elem
        
        # Handle numpy data types
        if isinstance(obj, np.integer):
            elem = ET.Element(element_name)
            elem.set("type", "numpy_int")
            elem.text = str(int(obj))
            return elem
        if isinstance(obj, np.floating):
            elem = ET.Element(element_name)
            elem.set("type", "numpy_float")
            elem.text = str(float(obj))
            return elem
        if isinstance(obj, np.bool_):
            elem = ET.Element(element_name)
            elem.set("type", "numpy_bool")
            elem.text = str(bool(obj))
            return elem
        
        # Handle other objects
        try:
            # Try to convert to dictionary if object has __dict__ attribute
            if hasattr(obj, "__dict__"):
                elem = ET.Element(element_name)
                elem.set("type", "object")
                elem.set("class", obj.__class__.__name__)
                encoder = CustomXMLEncoder(self.max_depth, self.current_depth + 1)
                for key, value in obj.__dict__.items():
                    key_name = self._sanitize_key(str(key))
                    child = encoder.encode(value, key_name)
                    child.set("key", str(key))
                    elem.append(child)
                return elem
            
            # For other objects, convert to string
            elem = ET.Element(element_name)
            elem.set("type", obj.__class__.__name__)
            elem.text = str(obj)
            return elem
        except:
            elem = ET.Element(element_name)
            elem.set("type", "unknown")
            elem.text = str(type(obj))
            return elem
    
    def _sanitize_key(self, key):
        """Sanitize dictionary key to be valid XML element name."""
        # Replace invalid characters with underscore
        sanitized = ""
        for char in key:
            if char.isalnum() or char == "_":
                sanitized += char
            else:
                sanitized += "_"
        
        # Ensure it starts with letter or underscore
        if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = "key_" + sanitized
        
        return sanitized


def pickle_to_xml(pickle_path, xml_path=None, indent=True, sort_keys=False, verbose=False, max_depth=100):
    """Convert a Pickle file to XML format."""
    # Default output path if not specified
    if xml_path is None:
        xml_path = os.path.splitext(pickle_path)[0] + '.xml'
    
    try:
        # Load data from Pickle file
        if verbose:
            print(f"Loading data from '{pickle_path}'...")
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if verbose:
            print(f"Data loaded successfully. Converting to XML...")
        
        # Convert data to XML
        encoder = CustomXMLEncoder(max_depth=max_depth)
        root = encoder.encode(data, "root")
        
        # Create XML tree
        tree = ET.ElementTree(root)
        
        # Write to file with pretty printing if requested
        if indent:
            # Pretty print the XML
            rough_string = ET.tostring(root, encoding='unicode')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Remove extra blank lines
            lines = [line for line in pretty_xml.split('\n') if line.strip()]
            pretty_xml = '\n'.join(lines)
            
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
        else:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
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
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Convert a Pickle file to XML format.')
    parser.add_argument('input', help='Path to the input Pickle file')
    parser.add_argument('output', nargs='?', help='Path to the output XML file (optional)')
    parser.add_argument('--no-indent', action='store_true', help='Do not pretty print XML (compact format)')
    parser.add_argument('--sort-keys', action='store_true', help='Sort dictionary keys')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information during conversion')
    parser.add_argument('--max-depth', type=int, default=100, help='Maximum recursion depth for nested objects')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Convert Pickle to XML
    pickle_to_xml(
        args.input, 
        args.output, 
        not args.no_indent, 
        args.sort_keys, 
        args.verbose, 
        args.max_depth
    )


if __name__ == '__main__':
    main()