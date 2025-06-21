#!/usr/bin/env python3
"""
Debug why the visualization fix didn't work as expected.
"""

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_actual_color_mapping():
    """Debug the actual color mapping with real data."""
    
    print("=== DEBUGGING ACTUAL COLOR MAPPING ===")
    
    # Load the actual test data
    data = np.load('/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver/data/output/test_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Focus on taxonomy_dissimilarity (feature index 3)
    feature_index = 3
    feature_name = "taxonomy_dissimilarity"
    feature_values = X_test[:, feature_index]
    
    print(f"Feature: {feature_name}")
    print(f"Feature values range: [{feature_values.min():.3f}, {feature_values.max():.3f}]")
    
    # Create DataFrame as done in visualization
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': y_test
    })
    
    print(f"Class distribution: {np.bincount(y_test.astype(int))}")
    
    # Replicate the EXACT logic from the fixed visualization
    unique_classes = sorted(df['Class'].unique())
    print(f"Unique classes: {unique_classes}")
    
    if len(unique_classes) == 2:
        # Determine which class has higher feature values
        mean_values = df.groupby('Class')['Feature Value'].mean()
        print(f"Mean values by class:\n{mean_values}")
        
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
        
        # Create legend labels: green = Match, red = Non-match
        legend_labels = []
        for class_val in sorted(unique_classes):
            if color_map[class_val] == 'red':
                legend_labels.append('Non-match')
            else:
                legend_labels.append('Match')
        
        print(f"Legend labels for sorted classes {sorted(unique_classes)}: {legend_labels}")
        
        # ANALYSIS
        print("\n=== ANALYSIS ===")
        print("Expected for taxonomy_dissimilarity:")
        print("- Class 1 (matches) should have LOW values and be GREEN")
        print("- Class 0 (non-matches) should have HIGH values and be RED")
        
        # Check what we actually got
        class_0_mean = mean_values[0.0]
        class_1_mean = mean_values[1.0]
        
        print(f"\nActual data:")
        print(f"- Class 0 (non-match) mean: {class_0_mean:.3f}")
        print(f"- Class 1 (match) mean: {class_1_mean:.3f}")
        
        if class_1_mean < class_0_mean:
            print("✓ Matches have lower dissimilarity than non-matches (CORRECT)")
            
            # Since class 1 has lower values, it should be low_value_class
            print(f"low_value_class should be 1.0, actual: {low_value_class}")
            print(f"high_value_class should be 0.0, actual: {high_value_class}")
            
            # For dissimilarity, low_value_class gets green, high_value_class gets red
            print(f"Class 1 (matches) gets color: {color_map.get(1.0, 'unknown')}")
            print(f"Class 0 (non-matches) gets color: {color_map.get(0.0, 'unknown')}")
            
            # Check legend
            print(f"Class 0 gets legend: {legend_labels[0] if len(legend_labels) > 0 else 'unknown'}")
            print(f"Class 1 gets legend: {legend_labels[1] if len(legend_labels) > 1 else 'unknown'}")
            
        else:
            print("✗ Matches do not have lower dissimilarity (WRONG)")

def debug_matplotlib_palette_ordering():
    """Debug if the issue is with matplotlib/seaborn palette ordering."""
    
    print("\n=== DEBUGGING MATPLOTLIB PALETTE ===")
    
    # Create simple test data
    df = pd.DataFrame({
        'Feature Value': [0.1, 0.2, 0.8, 0.9],
        'Class': [1.0, 1.0, 0.0, 0.0]  # Class 1 has low values, Class 0 has high values
    })
    
    print("Test data:")
    print(df)
    
    # Apply the logic
    mean_values = df.groupby('Class')['Feature Value'].mean()
    print(f"Mean values: {mean_values}")
    
    high_value_class = mean_values.idxmax()
    low_value_class = mean_values.idxmin()
    
    print(f"High value class: {high_value_class}")
    print(f"Low value class: {low_value_class}")
    
    # For dissimilarity
    color_map = {low_value_class: 'green', high_value_class: 'red'}
    print(f"Color map: {color_map}")
    
    # Test what happens with seaborn
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    
    # This is the key line from the visualization
    ax = sns.histplot(data=df, x='Feature Value', hue='Class', element='step', 
                     stat='density', common_norm=False, palette=color_map)
    
    # Check what colors were actually assigned
    legend = ax.get_legend()
    if legend:
        for i, text in enumerate(legend.get_texts()):
            print(f"Legend item {i}: '{text.get_text()}'")
            
        for i, patch in enumerate(legend.get_patches()):
            color = patch.get_facecolor()
            print(f"Legend patch {i}: color {color}")
    
    plt.close()
    
    print("\nThis might reveal if seaborn is reordering the palette...")

if __name__ == "__main__":
    debug_actual_color_mapping()
    debug_matplotlib_palette_ordering()