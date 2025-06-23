#!/usr/bin/env python3
"""
Temporary script to copy classifications from parallel_classifications.json
into training_dataset_2025-06-20.csv, adding setfit_prediction and is_parent_category columns.
"""

import json
import pandas as pd
from typing import Dict, Set

def load_classifications(json_path: str) -> Dict[str, Dict]:
    """Load classifications from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def determine_parent_categories(classifications: Dict[str, Dict]) -> Set[str]:
    """
    Determine which categories are parent categories by analyzing paths.
    A category is a parent if it appears before '>' in any path.
    """
    parent_categories = set()
    
    for person_data in classifications.values():
        paths = person_data.get('path', [])
        for path in paths:
            if '>' in path:
                # Split by '>' and take the parent (everything before the last '>')
                parts = [part.strip() for part in path.split('>')]
                if len(parts) > 1:
                    # All parts except the last are parents
                    parent_categories.update(parts[:-1])
    
    return parent_categories

def update_csv_with_classifications(csv_path: str, classifications: Dict[str, Dict], 
                                  parent_categories: Set[str]) -> pd.DataFrame:
    """
    Update CSV file with classifications from JSON.
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Initialize new columns
    df['setfit_prediction'] = None
    df['is_parent_category'] = None
    
    # Track statistics
    matched = 0
    unmatched = 0
    unmatched_ids = []
    
    # Process each row
    for idx, row in df.iterrows():
        person_id = row['personId']
        
        # Handle NaN values and convert to string
        if pd.isna(person_id):
            print(f"Warning: Row {idx} has NaN personId")
            unmatched += 1
            continue
            
        person_id = str(person_id)
        
        if person_id in classifications:
            # Get first label (as per user's previous instruction)
            labels = classifications[person_id].get('label', [])
            if labels:
                first_label = labels[0]
                df.at[idx, 'setfit_prediction'] = first_label
                df.at[idx, 'is_parent_category'] = first_label in parent_categories
                matched += 1
            else:
                print(f"Warning: PersonId {person_id} has no labels")
                unmatched += 1
                unmatched_ids.append(person_id)
        else:
            print(f"Warning: PersonId {person_id} not found in classifications")
            unmatched += 1
            unmatched_ids.append(person_id)
    
    print(f"\nResults:")
    print(f"  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")
    
    if unmatched_ids:
        print(f"\nFirst 10 unmatched personIds:")
        for pid in unmatched_ids[:10]:
            print(f"  {pid}")
        if len(unmatched_ids) > 10:
            print(f"  ... and {len(unmatched_ids) - 10} more")
    
    return df

def main():
    # File paths
    json_path = "data/input/parallel_classifications.json"
    csv_path = "data/input/training_dataset_classified_2025-06-21.csv"
    output_path = "data/input/training_dataset_classified_2025-06-21.csv"  # Overwrite original
    
    print("Loading classifications from JSON...")
    classifications = load_classifications(json_path)
    print(f"Loaded {len(classifications)} classifications")
    
    print("\nDetermining parent categories...")
    parent_categories = determine_parent_categories(classifications)
    print(f"Found {len(parent_categories)} parent categories:")
    for parent in sorted(parent_categories):
        print(f"  {parent}")
    
    print("\nUpdating CSV with classifications...")
    df = update_csv_with_classifications(csv_path, classifications, parent_categories)
    
    # Show sample results
    print("\nSample results:")
    sample_df = df[df['setfit_prediction'].notna()].head(10)
    for _, row in sample_df.iterrows():
        print(f"  {row['personId']}: {row['setfit_prediction']} (parent: {row['is_parent_category']})")
    
    # Show distribution
    print(f"\nClassification distribution:")
    if 'setfit_prediction' in df.columns:
        value_counts = df['setfit_prediction'].value_counts()
        for category, count in value_counts.head(10).items():
            is_parent = category in parent_categories if category else False
            print(f"  {category}: {count} (parent: {is_parent})")
    
    print(f"\nParent category distribution:")
    if 'is_parent_category' in df.columns:
        parent_counts = df['is_parent_category'].value_counts()
        print(f"  Child categories (False): {parent_counts.get(False, 0)}")
        print(f"  Parent categories (True): {parent_counts.get(True, 0)}")
        print(f"  Unclassified (NaN): {df['is_parent_category'].isna().sum()}")
    
    # Save updated CSV
    print(f"\nSaving updated CSV to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()