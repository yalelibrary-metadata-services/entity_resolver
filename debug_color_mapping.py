#!/usr/bin/env python3
"""
Debug script to understand the color mapping issue in visualization.

This script will help identify where the label mapping is going wrong.
"""

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_label_encoding():
    """Debug the label encoding process."""
    
    print("=== DEBUGGING LABEL ENCODING ===")
    
    # Simulate ground truth data format
    ground_truth_data = [
        ("id1", "id2", "true"),
        ("id3", "id4", "false"), 
        ("id5", "id6", "true"),
        ("id7", "id8", "false")
    ]
    
    print("Original ground truth data:")
    for left, right, label in ground_truth_data:
        print(f"  {left}, {right}, {label}")
    
    # Replicate the label conversion from feature_engineering.py line 1861-1862
    labels = np.array([1 if str(label).lower() == 'true' else 0 
                      for _, _, label in ground_truth_data])
    
    print(f"\nConverted labels: {labels}")
    print("Encoding: 'true' -> 1, 'false' -> 0")
    print("This means: 1 = Match, 0 = Non-match")
    
    return labels

def debug_feature_distribution_plotting():
    """Debug the feature distribution plotting logic."""
    
    print("\n=== DEBUGGING VISUALIZATION LOGIC ===")
    
    # Create sample data that mimics the actual problem
    # For cosine similarity: matches should have HIGH values, non-matches should have LOW values
    np.random.seed(42)
    
    # Simulate feature values where:
    # - Class 1 (matches) have higher values (0.6-1.0)
    # - Class 0 (non-matches) have lower values (0.0-0.5)
    match_values = np.random.uniform(0.6, 1.0, 100)  # Class 1 = matches
    non_match_values = np.random.uniform(0.0, 0.5, 100)  # Class 0 = non-matches
    
    # Combine the data
    feature_values = np.concatenate([match_values, non_match_values])
    class_labels = np.concatenate([np.ones(100), np.zeros(100)])  # 1=match, 0=non-match
    
    print(f"Class 0 (non-match) mean: {feature_values[class_labels == 0].mean():.3f}")
    print(f"Class 1 (match) mean: {feature_values[class_labels == 1].mean():.3f}")
    
    # Create DataFrame as done in visualization
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': class_labels
    })
    
    # Replicate the visualization logic from lines 69-75
    unique_classes = sorted(df['Class'].unique())
    print(f"Unique classes: {unique_classes}")
    
    if len(unique_classes) == 2:
        # Determine which class has higher feature values (should be matches)
        mean_values = df.groupby('Class')['Feature Value'].mean()
        print(f"Mean values by class:\n{mean_values}")
        
        high_value_class = mean_values.idxmax()  # Class with higher average feature values
        low_value_class = mean_values.idxmin()   # Class with lower average feature values
        
        print(f"High value class: {high_value_class}")
        print(f"Low value class: {low_value_class}")
        
        # This is the problematic assignment logic
        color_map = {high_value_class: 'green', low_value_class: 'red'}
        print(f"Color mapping: {color_map}")
        
        # Create legend labels based on color assignment
        legend_labels = []
        for class_val in sorted(unique_classes):
            if color_map[class_val] == 'red':
                legend_labels.append('Non-match')
            else:
                legend_labels.append('Match')
        
        print(f"Legend labels: {legend_labels}")
        
        # PROBLEM ANALYSIS
        print("\n=== PROBLEM ANALYSIS ===")
        print("Expected behavior:")
        print("  - Class 1 (matches) should be GREEN because they have high cosine similarity")
        print("  - Class 0 (non-matches) should be RED because they have low cosine similarity")
        
        print("\nActual behavior:")
        print(f"  - Class {high_value_class} gets GREEN (correctly, as it has higher values)")
        print(f"  - Class {low_value_class} gets RED (correctly, as it has lower values)")
        
        print("\nBut the LEGEND is wrong!")
        print("The legend is created by iterating through sorted(unique_classes) = [0, 1]")
        print("- Class 0 gets RED -> labeled as 'Non-match' ✓ CORRECT")
        print("- Class 1 gets GREEN -> labeled as 'Match' ✓ CORRECT")
        
        print("\nSo the logic should actually be working correctly...")
        print("Let me check if there's something else going on...")

def check_actual_data_distribution():
    """Check if the issue might be in the actual data distribution."""
    print("\n=== CHECKING POSSIBLE ISSUES ===")
    
    print("Possible causes of the color mapping issue:")
    print("1. The feature values might be inverted (low values for matches, high for non-matches)")
    print("2. The class labels might be swapped somewhere in the pipeline")
    print("3. The legend creation logic might have a bug")
    print("4. The data being passed to visualization might be different from training data")
    
    print("\nTo debug this properly, we need to:")
    print("1. Check the actual feature values and labels being passed to plot_feature_distributions")
    print("2. Verify that matches truly have higher cosine similarity values than non-matches")
    print("3. Add debug prints to the visualization function to see the actual mappings")

if __name__ == "__main__":
    debug_label_encoding()
    debug_feature_distribution_plotting()
    check_actual_data_distribution()