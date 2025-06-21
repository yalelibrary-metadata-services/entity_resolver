#!/usr/bin/env python3
"""
Script to regenerate visualization plots with the fixed color mapping.
"""

import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.append('src')

from visualization import plot_feature_distributions, plot_class_separation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def regenerate_visualization_plots():
    """Regenerate the visualization plots with corrected color mapping."""
    
    print("=== REGENERATING VISUALIZATION PLOTS ===")
    
    try:
        # Load the test data
        data = np.load('data/output/test_data.npz')
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"Loaded test data: {X_test.shape}")
        
        # Feature names
        feature_names = ['person_cosine', 'person_title_squared', 'composite_cosine', 
                        'taxonomy_dissimilarity', 'birth_death_match']
        
        output_dir = 'data/output'
        
        print("Regenerating feature distribution plots...")
        plot_feature_distributions(X_test, y_test, feature_names, output_dir)
        
        print("Regenerating class separation plots...")  
        plot_class_separation(X_test, y_test, feature_names, output_dir)
        
        print("✓ Successfully regenerated plots with corrected color mapping!")
        print("Check data/output/plots/feature_distributions/ for updated plots")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    regenerate_visualization_plots()