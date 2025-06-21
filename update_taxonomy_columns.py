#!/usr/bin/env python3
"""
Update taxonomy columns in training_dataset_classified.csv based on updated_identity_classification_map.json
"""

import pandas as pd
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_classification_map(json_path):
    """Load the identity classification map from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def determine_is_parent_category(label, taxonomy_path):
    """
    Determine if a label is a parent category by checking the taxonomy structure.
    """
    # Load taxonomy to get parent categories
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    # Extract parent categories from taxonomy
    parent_categories = set()
    if 'skos:hasTopConcept' in taxonomy:
        for concept in taxonomy['skos:hasTopConcept']:
            pref_label = concept.get('skos:prefLabel', {})
            parent_label = pref_label.get('@value', '') if isinstance(pref_label, dict) else ''
            if parent_label:
                parent_categories.add(parent_label)
    
    return label in parent_categories

def update_csv_with_classifications(csv_path, json_path, taxonomy_path, output_path):
    """
    Update the CSV file with new classifications from the JSON mapping.
    
    Args:
        csv_path: Path to the training dataset CSV
        json_path: Path to the identity classification JSON
        taxonomy_path: Path to the taxonomy JSON for parent category detection
        output_path: Path for the updated CSV file
    """
    logger.info(f"Loading CSV data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    logger.info(f"Loading classification map from {json_path}")
    classification_map = load_classification_map(json_path)
    logger.info(f"Loaded classifications for {len(classification_map)} identities")
    
    # Track updates
    updates_made = 0
    missing_identities = set()
    
    # Process each row
    for index, row in df.iterrows():
        identity = str(row['identity'])
        
        # Look up classification for this identity
        if identity in classification_map:
            identity_data = classification_map[identity]
            
            # Get the first label from the label array
            labels = identity_data.get('label', [])
            if labels:
                new_label = labels[0]  # Use the FIRST entry as requested
                
                # Determine if this is a parent category
                new_is_parent = determine_is_parent_category(new_label, taxonomy_path)
                
                # Update the row if different
                old_label = row['setfit_prediction']
                old_is_parent = row['is_parent_category']
                
                if old_label != new_label or old_is_parent != new_is_parent:
                    df.at[index, 'setfit_prediction'] = new_label
                    df.at[index, 'is_parent_category'] = new_is_parent
                    updates_made += 1
                    
                    logger.debug(f"Updated identity {identity}: '{old_label}' → '{new_label}', "
                               f"is_parent: {old_is_parent} → {new_is_parent}")
        else:
            missing_identities.add(identity)
    
    # Report results
    logger.info(f"Updates made: {updates_made}")
    if missing_identities:
        logger.warning(f"Missing identities in classification map: {sorted(missing_identities)}")
    
    # Save updated CSV
    logger.info(f"Saving updated CSV to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Update completed successfully")
    
    # Summary statistics
    label_counts = df['setfit_prediction'].value_counts()
    parent_count = df['is_parent_category'].sum()
    child_count = len(df) - parent_count
    
    logger.info(f"\nUpdated dataset summary:")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Parent categories: {parent_count}")
    logger.info(f"Child categories: {child_count}")
    logger.info(f"\nLabel distribution:")
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count}")

def main():
    """Main function to update taxonomy columns."""
    # File paths
    project_root = Path(__file__).parent
    csv_path = project_root / "data" / "input" / "training_dataset_classified.csv"
    json_path = project_root / "data" / "output" / "updated_identity_classification_map_2025-06-02.json"
    taxonomy_path = project_root / "data" / "input" / "taxonomy_revised.json"
    
    # Create backup
    backup_path = csv_path.with_suffix('.csv.backup')
    logger.info(f"Creating backup at {backup_path}")
    import shutil
    shutil.copy2(csv_path, backup_path)
    
    # Update the CSV file (in place)
    update_csv_with_classifications(csv_path, json_path, taxonomy_path, csv_path)

if __name__ == "__main__":
    main()