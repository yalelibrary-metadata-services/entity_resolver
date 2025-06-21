#!/usr/bin/env python3
"""
Script to regenerate feature distribution plots with corrected colors.
"""

import numpy as np
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.visualization import plot_feature_distributions

# Load test data
output_dir = "data/output"
test_data_path = os.path.join(output_dir, "test_data.npz")

if os.path.exists(test_data_path):
    data = np.load(test_data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load feature names from the training report
    import json
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    feature_names = report['feature_info']['effective_features']
    
    print(f"Regenerating plots for {len(feature_names)} features...")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution: {np.sum(y_test == 0)} non-matches, {np.sum(y_test == 1)} matches")
    
    # Generate feature distribution plots
    plot_feature_distributions(X_test, y_test, feature_names, output_dir)
    
    print("Feature distribution plots regenerated successfully!")
else:
    print(f"Test data not found at {test_data_path}")
    print("Please run the training pipeline first.")