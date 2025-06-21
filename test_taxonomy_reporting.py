#!/usr/bin/env python
"""
Test script to verify taxonomy feature appears in reporting outputs.
"""

import yaml
import logging
import numpy as np
from src.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_taxonomy_in_reports():
    """Verify the taxonomy feature appears in feature lists and reports."""
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create mock weaviate client and hash lookup
    mock_weaviate = None
    mock_hash_lookup = {
        '9.0#Agent100-15': {'person': 'hash1', 'composite': 'hash2'},
        '9.1#Agent700-22': {'person': 'hash3', 'composite': 'hash4'},
    }
    
    # Initialize feature engineering
    logger.info("Initializing feature engineering...")
    feature_eng = FeatureEngineering(config, mock_weaviate, mock_hash_lookup)
    
    # Check enabled features
    logger.info("\n" + "="*60)
    logger.info("ENABLED FEATURES CHECK")
    logger.info("="*60)
    
    if 'taxonomy_dissimilarity' in feature_eng.enabled_features:
        logger.info("✓ taxonomy_dissimilarity is in enabled_features list")
    else:
        logger.error("✗ taxonomy_dissimilarity is NOT in enabled_features list")
    
    # Check feature registry
    if 'taxonomy_dissimilarity' in feature_eng.feature_registry:
        logger.info("✓ taxonomy_dissimilarity is in feature_registry")
    else:
        logger.error("✗ taxonomy_dissimilarity is NOT in feature_registry")
    
    # Get feature names that would be used in reporting
    feature_names = feature_eng.get_feature_names()
    
    logger.info(f"\nTotal features: {len(feature_names)}")
    logger.info("Feature names for reporting:")
    for i, name in enumerate(feature_names):
        marker = "→" if name == 'taxonomy_dissimilarity' else " "
        logger.info(f"{marker} {i+1}. {name}")
    
    # Verify taxonomy feature is in the list
    if 'taxonomy_dissimilarity' in feature_names:
        logger.info("\n✓ SUCCESS: taxonomy_dissimilarity will appear in:")
        logger.info("  - Feature distribution plots")
        logger.info("  - ROC curves and AUC scores")
        logger.info("  - Feature importance charts")
        logger.info("  - Detailed test results CSV")
        logger.info("  - Feature weights visualization")
        logger.info("  - HTML reports")
    else:
        logger.error("\n✗ FAILURE: taxonomy_dissimilarity not found in feature names")
    
    # Test feature calculation to ensure it works
    logger.info("\n" + "="*60)
    logger.info("FEATURE CALCULATION TEST")
    logger.info("="*60)
    
    try:
        test_pairs = [('9.0#Agent100-15', '9.1#Agent700-22')]
        features = feature_eng.calculate_features(test_pairs)
        
        logger.info(f"Feature calculation successful")
        logger.info(f"Feature array shape: {features.shape}")
        
        # Find taxonomy feature index
        if 'taxonomy_dissimilarity' in feature_names:
            tax_idx = feature_names.index('taxonomy_dissimilarity')
            tax_value = features[0, tax_idx]
            logger.info(f"Taxonomy dissimilarity value: {tax_value:.3f}")
        
    except Exception as e:
        logger.error(f"Feature calculation failed: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION STATUS")
    logger.info("="*60)
    
    checks = {
        'Feature enabled': 'taxonomy_dissimilarity' in feature_eng.enabled_features,
        'In registry': 'taxonomy_dissimilarity' in feature_eng.feature_registry,
        'In feature names': 'taxonomy_dissimilarity' in feature_names,
        'Taxonomy initialized': feature_eng.taxonomy_dissimilarity is not None
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"{status} {check}")
    
    if all_passed:
        logger.info("\n✓ Taxonomy feature is fully integrated into reporting!")
    else:
        logger.error("\n✗ Some integration checks failed")


if __name__ == "__main__":
    test_taxonomy_in_reports()