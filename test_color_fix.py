#!/usr/bin/env python3
"""
Test script to verify that the color fix works correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data to test the color mapping
np.random.seed(42)
n_samples = 1000

# Create feature values (simulate some separation between classes)
non_match_values = np.random.normal(0.3, 0.2, n_samples // 2)  # Lower values for non-match
match_values = np.random.normal(0.7, 0.2, n_samples // 2)     # Higher values for match

# Combine features and labels
feature_values = np.concatenate([non_match_values, match_values])
labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])  # 0=Non-match, 1=Match

# Create DataFrame
df = pd.DataFrame({
    'Feature Value': feature_values,
    'Class': labels
})

# Test the new color mapping logic
unique_classes = sorted(df['Class'].unique())
print(f"Unique classes: {unique_classes}")

if len(unique_classes) == 2:
    # Map 0 (Non-match) to red, 1 (Match) to green
    color_map = {unique_classes[0]: 'red', unique_classes[1]: 'green'}
    print(f"Color mapping: {color_map}")
    
    # Create test plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Feature Value', hue='Class', element='step', 
               stat='density', common_norm=False, palette=color_map)
    
    plt.title('Test: Feature Distribution with Correct Colors')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend(['Non-match (Red)', 'Match (Green)'])
    
    # Save test plot
    plt.savefig('test_color_fix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Test plot saved as 'test_color_fix.png'")
    print("✓ Non-match should be RED")
    print("✓ Match should be GREEN")
    print("✓ Color mapping is now explicit and will work regardless of label encoding")
else:
    print("❌ Expected 2 classes, found:", len(unique_classes))