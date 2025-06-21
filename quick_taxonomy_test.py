#!/usr/bin/env python3
"""
Quick test to verify taxonomy feature functionality.
"""

import yaml
from src.taxonomy_feature import TaxonomyDissimilarity

# Load config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

print("Testing taxonomy feature...")

# Initialize taxonomy feature
taxonomy = TaxonomyDissimilarity(config)

print(f"Loaded {len(taxonomy.person_to_identity)} personId mappings")
print(f"Loaded {len(taxonomy.identity_categories)} identity classifications")

# Test with some sample personIds from the data
test_pairs = [
    ("772230#Agent100-15", "772230#Hub240-16-Agent"),  # Same identity
    ("772230#Agent100-15", "515773#Agent100-17"),      # Different domains
]

for pid1, pid2 in test_pairs:
    dissim = taxonomy.calculate_entity_dissimilarity(pid1, pid2)
    identity1 = taxonomy.person_to_identity.get(pid1)
    identity2 = taxonomy.person_to_identity.get(pid2)
    cats1 = taxonomy.identity_categories.get(identity1, set()) if identity1 else set()
    cats2 = taxonomy.identity_categories.get(identity2, set()) if identity2 else set()
    
    print(f"\nPair: {pid1} vs {pid2}")
    print(f"  Identities: {identity1} vs {identity2}")
    print(f"  Categories: {cats1} vs {cats2}")
    print(f"  Dissimilarity: {dissim}")

print("\nTaxonomy feature test completed!")