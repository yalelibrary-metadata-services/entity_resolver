#!/usr/bin/env python3
"""
Test script to verify taxonomy feature integration in the entity resolution pipeline.
"""

import os
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.taxonomy_feature import TaxonomyDissimilarity
from src.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_taxonomy_integration():
    """Test the taxonomy feature integration."""
    # Load config
    config_path = os.path.join(project_root, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load sample data
    training_data_path = config.get('classified_data_path', 'data/input/training_dataset_classified.csv')
    df = pd.read_csv(training_data_path)
    
    logger.info(f"Loaded {len(df)} records from training dataset")
    
    # Test 1: Verify taxonomy feature initialization
    logger.info("\n=== Test 1: Verify Taxonomy Feature Initialization ===")
    taxonomy_feature = TaxonomyDissimilarity(config)
    
    logger.info(f"Number of personId mappings: {len(taxonomy_feature.person_to_identity)}")
    logger.info(f"Number of identity classifications: {len(taxonomy_feature.identity_categories)}")
    logger.info(f"Number of parent categories: {len(taxonomy_feature.parent_child_map)}")
    
    # Test 2: Test dissimilarity calculations
    logger.info("\n=== Test 2: Test Dissimilarity Calculations ===")
    
    # Get some test cases from the data
    test_cases = [
        # Same identity (should be 0.0)
        ("772230#Agent100-15", "772230#Hub240-16-Agent"),  # Both are identity 9.0
        
        # Different identities in same domain
        ("786805#Agent100-17", "660801#Agent100-15"),  # Both are Music
        
        # Different domains
        ("53144#Agent700-22", "772230#Agent100-15"),  # Arts vs Music
        ("515773#Agent100-17", "772230#Agent100-15"),  # Religion vs Music
    ]
    
    for person_id1, person_id2 in test_cases:
        # Get identity info
        identity1 = taxonomy_feature.person_to_identity.get(person_id1)
        identity2 = taxonomy_feature.person_to_identity.get(person_id2)
        
        if identity1 and identity2:
            cats1 = taxonomy_feature.identity_categories.get(identity1, set())
            cats2 = taxonomy_feature.identity_categories.get(identity2, set())
            
            dissim = taxonomy_feature.calculate_entity_dissimilarity(person_id1, person_id2)
            
            logger.info(f"\nPair: {person_id1} vs {person_id2}")
            logger.info(f"  Identity 1: {identity1} - Categories: {cats1}")
            logger.info(f"  Identity 2: {identity2} - Categories: {cats2}")
            logger.info(f"  Dissimilarity: {dissim}")
    
    # Test 3: Feature engineering integration
    logger.info("\n=== Test 3: Feature Engineering Integration ===")
    
    # Check if taxonomy_dissimilarity is in enabled features
    enabled_features = config.get('features', {}).get('enabled', [])
    if 'taxonomy_dissimilarity' in enabled_features:
        logger.info("✓ taxonomy_dissimilarity is enabled in config")
    else:
        logger.warning("✗ taxonomy_dissimilarity is NOT enabled in config")
    
    # Test 4: Edge cases
    logger.info("\n=== Test 4: Edge Cases ===")
    
    # Test with unknown personIds
    unknown_dissim = taxonomy_feature.calculate_entity_dissimilarity("unknown1", "unknown2")
    logger.info(f"Unknown personIds dissimilarity: {unknown_dissim} (should be 0.5)")
    
    # Test multi-domain entity (identity 9.0 has both parent and child categories)
    identity_9_cats = taxonomy_feature.identity_categories.get("9.0", set())
    logger.info(f"\nIdentity 9.0 categories: {identity_9_cats}")
    
    # Find some records for identity 9.0
    identity_9_records = df[df['identity'] == 9.0]['personId'].tolist()[:3]
    if len(identity_9_records) >= 2:
        dissim_same = taxonomy_feature.calculate_entity_dissimilarity(
            identity_9_records[0], identity_9_records[1]
        )
        logger.info(f"Same multi-domain identity dissimilarity: {dissim_same} (should be 0.0)")
    
    logger.info("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    test_taxonomy_integration()