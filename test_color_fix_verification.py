#!/usr/bin/env python3
"""
Test script to verify the color mapping fix works correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile

def test_dissimilarity_feature_color_mapping():
    """Test that dissimilarity features get correct color mapping."""
    
    print("=== TESTING DISSIMILARITY FEATURE COLOR MAPPING ===")
    
    # Create sample data that mimics taxonomy_dissimilarity
    np.random.seed(42)
    
    # For dissimilarity: matches should have LOW values, non-matches should have HIGH values
    match_values = np.random.uniform(0.0, 0.1, 100)     # Class 1 = matches (low dissimilarity)
    non_match_values = np.random.uniform(0.2, 0.4, 100) # Class 0 = non-matches (high dissimilarity)
    
    # Combine the data
    feature_values = np.concatenate([match_values, non_match_values])
    class_labels = np.concatenate([np.ones(100), np.zeros(100)])  # 1=match, 0=non-match
    
    print(f"Class 0 (non-match) mean: {feature_values[class_labels == 0].mean():.3f}")
    print(f"Class 1 (match) mean: {feature_values[class_labels == 1].mean():.3f}")
    print("Expected: Non-matches should have HIGHER values for dissimilarity")
    
    # Create DataFrame as done in visualization
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': class_labels
    })
    
    # Apply the NEW logic for dissimilarity features
    feature_name = "taxonomy_dissimilarity"  # This should trigger dissimilarity logic
    
    unique_classes = sorted(df['Class'].unique())
    if len(unique_classes) == 2:
        # Determine which class has higher feature values
        mean_values = df.groupby('Class')['Feature Value'].mean()
        high_value_class = mean_values.idxmax()  # Class with higher average feature values
        low_value_class = mean_values.idxmin()   # Class with lower average feature values
        
        print(f"High value class: {high_value_class}")
        print(f"Low value class: {low_value_class}")
        
        # NEW LOGIC: Check if it's a dissimilarity feature
        is_dissimilarity_feature = 'dissimilarity' in feature_name.lower() or 'distance' in feature_name.lower()
        print(f"Is dissimilarity feature: {is_dissimilarity_feature}")
        
        if is_dissimilarity_feature:
            # For dissimilarity: low values = matches = green, high values = non-matches = red
            color_map = {low_value_class: 'green', high_value_class: 'red'}
            print("Using dissimilarity logic: LOW values = GREEN (matches)")
        else:
            # For similarity: high values = matches = green, low values = non-matches = red
            color_map = {high_value_class: 'green', low_value_class: 'red'}
            print("Using similarity logic: HIGH values = GREEN (matches)")
        
        print(f"Color mapping: {color_map}")
        
        # Create legend labels: green = Match, red = Non-match
        legend_labels = []
        for class_val in sorted(unique_classes):
            if color_map[class_val] == 'red':
                legend_labels.append('Non-match')
            else:
                legend_labels.append('Match')
        
        print(f"Legend labels: {legend_labels}")
        
        # Verify the mapping is correct
        print("\n=== VERIFICATION ===")
        if low_value_class == 1.0:  # Class 1 = matches
            print("✓ Matches have low dissimilarity values (CORRECT)")
            if color_map[1.0] == 'green':
                print("✓ Matches are colored GREEN (CORRECT)")
            else:
                print("✗ Matches are colored RED (WRONG)")
        else:
            print("✗ Matches do not have low dissimilarity values (WRONG)")
            
        if high_value_class == 0.0:  # Class 0 = non-matches
            print("✓ Non-matches have high dissimilarity values (CORRECT)")
            if color_map[0.0] == 'red':
                print("✓ Non-matches are colored RED (CORRECT)")
            else:
                print("✗ Non-matches are colored GREEN (WRONG)")
        else:
            print("✗ Non-matches do not have high dissimilarity values (WRONG)")
        
        # Create a test plot
        with tempfile.TemporaryDirectory() as temp_dir:
            plt.figure(figsize=(10, 6))
            
            sns.histplot(data=df, x='Feature Value', hue='Class', element='step', 
                       stat='density', common_norm=False, palette=color_map)
            
            plt.title(f'Test Plot: {feature_name}')
            plt.xlabel('Feature Value')
            plt.ylabel('Density')
            plt.legend(legend_labels)
            
            plot_path = os.path.join(temp_dir, 'test_dissimilarity_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nTest plot saved to: {plot_path}")
            print("In the plot, you should see:")
            print("- GREEN distribution at LOW values (matches)")
            print("- RED distribution at HIGH values (non-matches)")

def test_similarity_feature_color_mapping():
    """Test that similarity features still get correct color mapping."""
    
    print("\n=== TESTING SIMILARITY FEATURE COLOR MAPPING ===")
    
    # Create sample data that mimics person_cosine
    np.random.seed(42)
    
    # For similarity: matches should have HIGH values, non-matches should have LOW values
    match_values = np.random.uniform(0.7, 1.0, 100)     # Class 1 = matches (high similarity)
    non_match_values = np.random.uniform(0.3, 0.6, 100) # Class 0 = non-matches (low similarity)
    
    # Combine the data
    feature_values = np.concatenate([match_values, non_match_values])
    class_labels = np.concatenate([np.ones(100), np.zeros(100)])  # 1=match, 0=non-match
    
    print(f"Class 0 (non-match) mean: {feature_values[class_labels == 0].mean():.3f}")
    print(f"Class 1 (match) mean: {feature_values[class_labels == 1].mean():.3f}")
    print("Expected: Matches should have HIGHER values for similarity")
    
    # Create DataFrame as done in visualization
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': class_labels
    })
    
    # Apply the logic for similarity features
    feature_name = "person_cosine"  # This should NOT trigger dissimilarity logic
    
    unique_classes = sorted(df['Class'].unique())
    if len(unique_classes) == 2:
        # Determine which class has higher feature values
        mean_values = df.groupby('Class')['Feature Value'].mean()
        high_value_class = mean_values.idxmax()  # Class with higher average feature values
        low_value_class = mean_values.idxmin()   # Class with lower average feature values
        
        print(f"High value class: {high_value_class}")
        print(f"Low value class: {low_value_class}")
        
        # Check if it's a dissimilarity feature
        is_dissimilarity_feature = 'dissimilarity' in feature_name.lower() or 'distance' in feature_name.lower()
        print(f"Is dissimilarity feature: {is_dissimilarity_feature}")
        
        if is_dissimilarity_feature:
            # For dissimilarity: low values = matches = green, high values = non-matches = red
            color_map = {low_value_class: 'green', high_value_class: 'red'}
            print("Using dissimilarity logic: LOW values = GREEN (matches)")
        else:
            # For similarity: high values = matches = green, low values = non-matches = red
            color_map = {high_value_class: 'green', low_value_class: 'red'}
            print("Using similarity logic: HIGH values = GREEN (matches)")
        
        print(f"Color mapping: {color_map}")
        
        # Verify the mapping is correct
        print("\n=== VERIFICATION ===")
        if high_value_class == 1.0:  # Class 1 = matches
            print("✓ Matches have high similarity values (CORRECT)")
            if color_map[1.0] == 'green':
                print("✓ Matches are colored GREEN (CORRECT)")
            else:
                print("✗ Matches are colored RED (WRONG)")
        else:
            print("✗ Matches do not have high similarity values (WRONG)")

if __name__ == "__main__":
    test_dissimilarity_feature_color_mapping()
    test_similarity_feature_color_mapping()