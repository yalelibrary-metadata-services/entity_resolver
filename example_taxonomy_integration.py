#!/usr/bin/env python
"""
Example of using the taxonomy dissimilarity feature in the entity resolution pipeline.

This demonstrates how the new feature helps differentiate between entities with
similar names but different professional domains.
"""

import yaml
import logging
from src.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_taxonomy_feature():
    """Demonstrate how the taxonomy feature works in practice."""
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create mock weaviate client and hash lookup for demo
    # In real usage, these would come from your pipeline
    mock_weaviate = None  # Placeholder
    mock_hash_lookup = {
        '9.0#Agent100-15': {'person': 'hash1', 'composite': 'hash2'},
        '9.1#Agent700-22': {'person': 'hash3', 'composite': 'hash4'},
        '54.0#Agent100-17': {'person': 'hash5', 'composite': 'hash6'},
    }
    
    # Initialize feature engineering with taxonomy support
    logger.info("Initializing feature engineering with taxonomy support...")
    feature_eng = FeatureEngineering(config, mock_weaviate, mock_hash_lookup)
    
    # Check if taxonomy feature is enabled
    if 'taxonomy_dissimilarity' in feature_eng.feature_names:
        logger.info("✓ Taxonomy dissimilarity feature is enabled")
    else:
        logger.warning("✗ Taxonomy dissimilarity feature is not enabled")
        return
    
    # Example entity pairs
    pairs = [
        {
            'name': 'Same person (composer Schubert)',
            'pair': ('9.0#Agent100-15', '9.0#Agent100-15'),
            'expected': 'Low dissimilarity (0.0)'
        },
        {
            'name': 'Different Schuberts (composer vs archaeologist)',
            'pair': ('9.0#Agent100-15', '9.1#Agent700-22'),
            'expected': 'Moderate dissimilarity (0.3 - sibling categories)'
        },
        {
            'name': 'Unrelated people (Schubert vs Strauss)',
            'pair': ('9.0#Agent100-15', '54.0#Agent100-17'),
            'expected': 'Low dissimilarity (0.0 - both composers)'
        }
    ]
    
    logger.info("\n" + "="*60)
    logger.info("FEATURE CALCULATIONS")
    logger.info("="*60)
    
    for example in pairs:
        logger.info(f"\nExample: {example['name']}")
        logger.info(f"Comparing: {example['pair'][0]} vs {example['pair'][1]}")
        logger.info(f"Expected: {example['expected']}")
        
        # Calculate all features for this pair
        features = feature_eng.compute_features_for_pair(
            example['pair'][0], 
            example['pair'][1]
        )
        
        # Show taxonomy dissimilarity value
        if 'taxonomy_dissimilarity' in features:
            tax_value = features['taxonomy_dissimilarity']
            logger.info(f"Taxonomy dissimilarity: {tax_value:.3f}")
            
            # Interpret the value
            if tax_value < 0.1:
                logger.info("→ Same professional domain")
            elif tax_value < 0.4:
                logger.info("→ Related domains (sibling categories)")
            elif tax_value < 0.7:
                logger.info("→ Somewhat related domains")
            else:
                logger.info("→ Unrelated domains")
        
        # Show other features for comparison
        logger.info("\nOther features:")
        for feat_name, feat_value in features.items():
            if feat_name != 'taxonomy_dissimilarity':
                logger.info(f"  {feat_name}: {feat_value:.3f}")
    
    # Show feature importance
    logger.info("\n" + "="*60)
    logger.info("FEATURE CONFIGURATION")
    logger.info("="*60)
    
    logger.info("\nEnabled features:")
    for i, feat_name in enumerate(feature_eng.feature_names):
        weight = config['features']['parameters'].get(feat_name, {}).get('weight', 1.0)
        logger.info(f"  {i+1}. {feat_name} (weight: {weight})")
    
    # Explain how taxonomy helps
    logger.info("\n" + "="*60)
    logger.info("HOW TAXONOMY DISSIMILARITY HELPS")
    logger.info("="*60)
    
    logger.info("""
The taxonomy dissimilarity feature helps differentiate entities by:

1. **Domain Distinction**: Clearly separates people from different professional 
   domains (e.g., composers vs archaeologists) even with identical names.

2. **Hierarchical Relationships**: Uses the 2-level taxonomy structure:
   - Same category: 0.0 (strong match indicator)
   - Sibling categories: 0.3 (same parent domain)
   - Different parents: 0.8 (unrelated domains)

3. **Multi-Domain Handling**: For people working across domains, uses the 
   minimum dissimilarity to avoid false negatives.

4. **Edge Case Handling**:
   - Missing classifications: Returns 0.5 (neutral)
   - Unknown categories: Returns 0.9 (high dissimilarity)

This feature is particularly valuable when other features (name similarity, 
title similarity) might incorrectly suggest a match between different people.
""")


if __name__ == "__main__":
    demonstrate_taxonomy_feature()