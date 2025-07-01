#!/usr/bin/env python3
"""
Feature Weights Bar Chart
Clean horizontal bar chart showing positive and negative weights
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_weights_bar():
    """Create clean bar chart for feature weights with direction"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Feature data (from real Yale pipeline)
    features = [
        'Person Similarity',
        'Full Record Similarity', 
        'Person-Title Interaction',
        'Birth-Death Match',
        'Domain Difference'
    ]
    
    # Real logistic regression weights from Yale's feature_weight_info.json
    weights = [0.603, 1.458, 1.017, 2.514, -1.812]
    
    # Reorder for better visual flow (positive first, then negative)
    sorted_data = sorted(zip(features, weights), key=lambda x: x[1], reverse=True)
    sorted_features, sorted_weights = zip(*sorted_data)
    
    # High contrast colors for maximum accessibility
    # Using dark green and dark red for better contrast against white background
    colors = ['#006400' if w > 0 else '#8B0000' for w in sorted_weights]  # DarkGreen and DarkRed
    
    # Create horizontal bars
    y_positions = range(len(sorted_features))
    bars = ax.barh(y_positions, sorted_weights, color=colors, alpha=1.0,  # Full opacity for max contrast
                   edgecolor='black', linewidth=2, height=0.6)  # Black edge for definition
    
    # Add value labels with precise positioning to avoid overlap
    for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
        width = bar.get_width()
        
        # Adjust positioning based on value to prevent going off chart
        if width > 0:
            if width > 2.0:  # For very large values like 2.51
                label_x = width - 0.15  # Position inside the bar
                ha = 'right'
                color = 'black'  # Changed from white to black for better readability
            else:
                label_x = width + 0.05  # Position just outside
                ha = 'left'
                color = 'black'
        else:
            label_x = width - 0.05  # Position just to the left for negative values
            ha = 'right'
            color = 'black'
            
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{weight:.2f}',
                ha=ha, va='center', 
                fontweight='bold', fontsize=13, color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.9))
    
    # Customize axes with better spacing
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_features, fontsize=13, fontweight='bold')
    ax.set_xlabel('Logistic Regression Weight', fontweight='bold', fontsize=15)
    ax.set_title('Feature Weights: Impact on Match Probability\n(Positive = Increases | Negative = Decreases)', 
                 fontweight='bold', fontsize=17, pad=25)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
    
    # Enhanced grid
    ax.grid(axis='x', alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)
    
    # Add interpretation boxes with high contrast colors matching the bars
    ax.text(1.2, 3.8, '✅ POSITIVE WEIGHTS\nIncrease match probability\nStronger signal = more likely same person', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#006400',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E6F5E6', 
                     edgecolor='#006400', linewidth=2))
    
    ax.text(-1.2, 1.2, '❌ NEGATIVE WEIGHTS\nDecrease match probability\nStronger signal = more likely different people', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#8B0000',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE6E6', 
                     edgecolor='#8B0000', linewidth=2))
    
    # Adjust layout with wider x-limits to prevent cutoff and accommodate labels
    ax.set_xlim(-2.2, 2.8)  # Extended right side for "2.51" label
    plt.tight_layout(pad=2.0)
    
    return fig

def main():
    """Generate and save the weights bar chart"""
    
    fig = create_weights_bar()
    
    output_path = Path(__file__).parent / "img" / "feature_weights_bar.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Feature weights bar chart saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()