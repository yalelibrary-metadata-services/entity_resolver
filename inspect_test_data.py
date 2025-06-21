#!/usr/bin/env python3
"""
Inspect the actual test data to understand the feature distributions.
"""

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_test_data():
    """Load and inspect the actual test data."""
    
    print("=== INSPECTING ACTUAL TEST DATA ===")
    
    try:
        # Load the test data
        data = np.load('/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver/data/output/test_data.npz')
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Unique labels: {np.unique(y_test)}")
        print(f"Label distribution: {np.bincount(y_test.astype(int))}")
        
        # Check each feature
        feature_names = ['person_cosine', 'person_title_squared', 'composite_cosine', 
                        'taxonomy_dissimilarity', 'birth_death_match']
        
        for i, feature_name in enumerate(feature_names):
            if i < X_test.shape[1]:
                feature_values = X_test[:, i]
                
                print(f"\n--- {feature_name} ---")
                print(f"Overall range: [{feature_values.min():.3f}, {feature_values.max():.3f}]")
                
                # Split by class
                match_values = feature_values[y_test == 1]
                non_match_values = feature_values[y_test == 0]
                
                print(f"Match (class 1) values:")
                print(f"  Count: {len(match_values)}")
                print(f"  Mean: {match_values.mean():.3f}")
                print(f"  Range: [{match_values.min():.3f}, {match_values.max():.3f}]")
                
                print(f"Non-match (class 0) values:")
                print(f"  Count: {len(non_match_values)}")
                print(f"  Mean: {non_match_values.mean():.3f}")
                print(f"  Range: [{non_match_values.min():.3f}, {non_match_values.max():.3f}]")
                
                # Analyze the relationship
                if match_values.mean() > non_match_values.mean():
                    print(f"  ✓ Matches have HIGHER values (expected for similarity features)")
                    print(f"  Expected visualization: Matches=GREEN, Non-matches=RED")
                else:
                    print(f"  ⚠ Matches have LOWER values (unexpected for similarity features!)")
                    print(f"  Expected visualization: Matches=RED, Non-matches=GREEN")
                    print(f"  This would explain the color mapping issue!")
                
    except Exception as e:
        print(f"Error loading test data: {e}")
        
        # Try to load from a CSV file instead
        print("\nTrying to load from CSV files...")
        try:
            # Look for recent test results
            import glob
            import os
            
            csv_files = glob.glob('/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver/data/output/detailed_test_results_*.csv')
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getctime)
                print(f"Loading from: {latest_csv}")
                
                df = pd.read_csv(latest_csv)
                print(f"CSV columns: {list(df.columns)}")
                
                # Look for feature columns
                feature_cols = [col for col in df.columns if any(feat in col.lower() for feat in ['cosine', 'similarity', 'birth_death', 'taxonomy'])]
                print(f"Feature columns found: {feature_cols}")
                
                if 'actual_label' in df.columns and feature_cols:
                    print("\nAnalyzing features from CSV:")
                    for col in feature_cols[:3]:  # Just check first 3 features
                        print(f"\n--- {col} ---")
                        
                        matches = df[df['actual_label'] == 1][col]
                        non_matches = df[df['actual_label'] == 0][col]
                        
                        print(f"Matches mean: {matches.mean():.3f}")
                        print(f"Non-matches mean: {non_matches.mean():.3f}")
                        
                        if matches.mean() > non_matches.mean():
                            print("✓ Matches have higher values")
                        else:
                            print("⚠ Matches have lower values")
            
        except Exception as e2:
            print(f"Error loading CSV data: {e2}")

if __name__ == "__main__":
    inspect_test_data()