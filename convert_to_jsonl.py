#!/usr/bin/env python3
"""
Convert yale-library-entity-resolver-classifications.json to JSON Lines format.
Each entity will become a separate line in the output file.
"""

import json
import sys
from pathlib import Path

def convert_json_to_jsonl(input_file: str, output_file: str):
    """
    Convert JSON file to JSON Lines format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    try:
        # Read the input JSON file
        print(f"Reading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded JSON with {len(data)} entities")
        
        # Write to JSON Lines format
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entity_id, entity_data in data.items():
                # Create a record with the entity ID and its data
                record = {
                    "entity_id": entity_id,
                    **entity_data
                }
                # Write each record as a single line
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Successfully converted {len(data)} entities to JSON Lines format")
        
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        return False
    
    return True

if __name__ == "__main__":
    input_file = "data/input/yale-library-entity-resolver-classifications.json"
    output_file = "data/input/yale-library-entity-resolver-classifications.jsonl"
    
    # Convert the file
    success = convert_json_to_jsonl(input_file, output_file)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)