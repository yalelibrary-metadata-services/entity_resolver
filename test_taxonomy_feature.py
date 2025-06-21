#!/usr/bin/env python
"""
Test script for the taxonomy dissimilarity feature.

This script validates the taxonomy feature implementation and demonstrates
how it handles various edge cases including multi-domain entities.
"""

import sys
import logging
import yaml
from src.taxonomy_feature import TaxonomyDissimilarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_taxonomy_feature():
    """Test the taxonomy dissimilarity feature with various scenarios."""
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize taxonomy dissimilarity
    logger.info("Initializing taxonomy dissimilarity calculator...")
    tax_calc = TaxonomyDissimilarity(config)
    
    # Test cases
    test_cases = [
        # Same domain - composers
        {
            'name': 'Two composers (Schubert)',
            'id1': '9.0',  # Franz Schubert composer
            'id2': '9.0',  # Same entity
            'expected': 0.0,
            'description': 'Same entity should have 0.0 dissimilarity'
        },
        {
            'name': 'Composer vs Archaeologist (both Schubert)',
            'id1': '9.0',  # Franz Schubert composer (Music)
            'id2': '9.1',  # Franz Schubert archaeologist (Arts/Visual)
            'expected': 0.4,  # Parent-child relationship after filtering
            'description': 'Different Schuberts - Music (child) vs Arts (parent) = parent-child dissimilarity'
        },
        {
            'name': 'Composer vs Religious figure',
            'id1': '9.0',  # Franz Schubert composer
            'id2': '49.0', # John Wesley religious figure
            'expected': 0.8,  # Different parent categories
            'description': 'Different parent categories (Arts vs Humanities)'
        },
        {
            'name': 'Literature figures',
            'id1': '58.0', # James Laughlin (Literature)
            'id2': '59.0', # James Joyce (Literature)
            'expected': 0.0,  # Same category
            'description': 'Both in Literature category'
        }
    ]
    
    # Run tests
    logger.info("\n" + "="*60)
    logger.info("TESTING TAXONOMY DISSIMILARITY FEATURE")
    logger.info("="*60 + "\n")
    
    for test in test_cases:
        logger.info(f"Test: {test['name']}")
        logger.info(f"Description: {test['description']}")
        
        # Get entity categories for debugging
        id1_info = tax_calc.get_debug_info(test['id1'])
        id2_info = tax_calc.get_debug_info(test['id2'])
        
        logger.info(f"Entity 1 ({test['id1']}): {id1_info.get('categories', [])}")
        logger.info(f"Entity 2 ({test['id2']}): {id2_info.get('categories', [])}")
        
        # Calculate dissimilarity
        dissim = tax_calc.calculate_entity_dissimilarity(test['id1'], test['id2'])
        
        logger.info(f"Calculated dissimilarity: {dissim:.2f}")
        logger.info(f"Expected dissimilarity: {test['expected']:.2f}")
        
        # Check if close to expected (allowing small tolerance)
        if abs(dissim - test['expected']) < 0.01:
            logger.info("✓ PASSED")
        else:
            logger.warning("✗ FAILED")
            
        logger.info("-" * 40)
    
    # Test multi-domain entity handling
    logger.info("\n" + "="*60)
    logger.info("TESTING MULTI-DOMAIN ENTITY HANDLING")
    logger.info("="*60 + "\n")
    
    # Look for entities with multiple categories
    multi_domain_entities = []
    for entity_id, categories in tax_calc.entity_categories.items():
        if len(categories) > 1:
            multi_domain_entities.append((entity_id, categories))
    
    if multi_domain_entities:
        logger.info(f"Found {len(multi_domain_entities)} multi-domain entities")
        
        # Show first few examples
        for entity_id, categories in multi_domain_entities[:5]:
            logger.info(f"\nEntity {entity_id} has {len(categories)} categories:")
            for cat in categories:
                parent = tax_calc.get_parent_category(cat)
                logger.info(f"  - {cat} (parent: {parent})")
                
        # Test dissimilarity with a multi-domain entity
        if multi_domain_entities:
            multi_id = multi_domain_entities[0][0]
            single_id = '9.0'  # A single-category entity
            
            logger.info(f"\nTesting multi-domain entity {multi_id} vs single-domain {single_id}")
            dissim = tax_calc.calculate_entity_dissimilarity(multi_id, single_id)
            logger.info(f"Dissimilarity: {dissim:.2f}")
            logger.info("(Should use minimum dissimilarity across all category pairs)")
    else:
        logger.info("No multi-domain entities found in the dataset")
    
    # Test edge cases
    logger.info("\n" + "="*60)
    logger.info("TESTING EDGE CASES")
    logger.info("="*60 + "\n")
    
    # Test with non-existent entity
    logger.info("Test: Non-existent entity")
    dissim = tax_calc.calculate_entity_dissimilarity('999999.0', '9.0')
    logger.info(f"Dissimilarity with non-existent entity: {dissim:.2f}")
    logger.info(f"(Should be 0.5 - neutral value for missing data)")
    
    # Test category hierarchy
    logger.info("\n" + "="*60)
    logger.info("TAXONOMY STRUCTURE")
    logger.info("="*60 + "\n")
    
    logger.info("Parent categories and their children:")
    for parent, children in tax_calc.parent_child_map.items():
        logger.info(f"\n{parent}:")
        for child in sorted(children):
            logger.info(f"  - {child}")
        # if len(children) > 5:
        #     logger.info(f"  ... and {len(children) - 5} more")


if __name__ == "__main__":
    test_taxonomy_feature()