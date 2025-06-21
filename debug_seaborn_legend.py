#!/usr/bin/env python3
"""
Debug seaborn legend creation to understand what's happening.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def debug_seaborn_legend_creation():
    """Debug exactly how seaborn creates legends and assigns colors."""
    
    print("=== DEBUGGING SEABORN LEGEND CREATION ===")
    
    # Load actual data
    data = np.load('/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver/data/output/test_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Get taxonomy_dissimilarity feature
    feature_values = X_test[:, 3]
    
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': y_test
    })
    
    print("Sample data:")
    print(df.head(10))
    print(f"Unique classes: {sorted(df['Class'].unique())}")
    
    # Analyze means
    mean_values = df.groupby('Class')['Feature Value'].mean()
    print(f"Mean values by class:\n{mean_values}")
    
    high_value_class = mean_values.idxmax()
    low_value_class = mean_values.idxmin()
    
    print(f"High value class: {high_value_class} (should get one color)")
    print(f"Low value class: {low_value_class} (should get another color)")
    
    # Create the color map as in our fixed code
    color_map = {low_value_class: 'green', high_value_class: 'red'}
    print(f"Our color map: {color_map}")
    
    # Create the plot and capture what seaborn actually does
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=df, x='Feature Value', hue='Class', multiple='dodge', 
                 stat='density', common_norm=False, palette=color_map, ax=ax)
    
    # Examine the legend
    legend = ax.get_legend()
    if legend:
        print("\nLegend analysis:")
        legend_labels = [text.get_text() for text in legend.get_texts()]
        legend_colors = [patch.get_facecolor() for patch in legend.get_patches()]
        
        for i, (label, color) in enumerate(zip(legend_labels, legend_colors)):
            color_name = "RED" if color[0] > 0.5 else "GREEN" if color[1] > 0.5 else "OTHER"
            print(f"  Legend item {i}: Class {label} -> {color_name} color {color}")
    
    # Also check the bars/patches directly
    print("\nBar analysis:")
    for i, patch in enumerate(ax.patches):
        color = patch.get_facecolor()
        x_pos = patch.get_x()
        width = patch.get_width()
        height = patch.get_height()
        color_name = "RED" if color[0] > 0.5 else "GREEN" if color[1] > 0.5 else "OTHER"
        
        if height > 0.1:  # Only show significant bars
            print(f"  Bar {i}: x={x_pos:.3f}, width={width:.3f}, height={height:.1f}, color={color_name}")
    
    plt.title("Debug: taxonomy_dissimilarity")
    plt.close()
    
    # Test if the issue is in our legend label creation
    print("\nTesting our legend label creation:")
    unique_classes = sorted(df['Class'].unique())
    print(f"Sorted unique classes: {unique_classes}")
    
    legend_labels = []
    for class_val in sorted(unique_classes):
        if color_map[class_val] == 'red':
            legend_labels.append('Non-match')
        else:
            legend_labels.append('Match')
    
    print(f"Our legend labels: {legend_labels}")
    print("This means:")
    for i, class_val in enumerate(sorted(unique_classes)):
        color = color_map[class_val]
        label = legend_labels[i]
        print(f"  Class {class_val} -> {color} -> '{label}'")

def test_explicit_legend_override():
    """Test if we can explicitly override the legend."""
    
    print("\n=== TESTING EXPLICIT LEGEND OVERRIDE ===")
    
    # Load actual data
    data = np.load('/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver/data/output/test_data.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Get taxonomy_dissimilarity feature
    feature_values = X_test[:, 3]
    
    df = pd.DataFrame({
        'Feature Value': feature_values,
        'Class': y_test
    })
    
    # Analyze
    mean_values = df.groupby('Class')['Feature Value'].mean()
    high_value_class = mean_values.idxmax()
    low_value_class = mean_values.idxmin()
    
    color_map = {low_value_class: 'green', high_value_class: 'red'}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=df, x='Feature Value', hue='Class', multiple='dodge', 
                 stat='density', common_norm=False, palette=color_map, ax=ax)
    
    # EXPLICITLY set the legend
    handles, labels = ax.get_legend_handles_labels()
    print(f"Original legend handles: {len(handles)}")
    print(f"Original legend labels: {labels}")
    
    # Create our own legend
    # We know class order from sorted(unique_classes)
    new_labels = []
    for class_val in sorted(df['Class'].unique()):
        if class_val == 1:  # Class 1 = matches
            new_labels.append('Match')
        else:  # Class 0 = non-matches
            new_labels.append('Non-match')
    
    print(f"New legend labels: {new_labels}")
    
    # Set the legend with our labels
    ax.legend(handles, new_labels)
    
    plt.title("Test: Explicit Legend Override")
    plt.savefig('/tmp/debug_explicit_legend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved test plot to /tmp/debug_explicit_legend.png")

if __name__ == "__main__":
    debug_seaborn_legend_creation()
    test_explicit_legend_override()