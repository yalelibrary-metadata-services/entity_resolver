#!/usr/bin/env python3
"""
Clean Feature Importance Radar Chart
Simplified version focusing on the radar visualization only
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_clean_radar():
    """Create clean, simple radar chart for feature importance"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    # Feature data (from real Yale pipeline) - simplified labels to prevent overlap
    features = [
        'Person\nSimilarity',
        'Record\nSimilarity', 
        'Person-Title\nInteraction',
        'Domain\nDifference',
        'Birth-Death\nMatch'
    ]
    
    # Real importance values from Yale's feature_weight_info.json (normalized, absolute values)
    raw_weights = [0.603, 1.458, 1.017, 1.812, 2.514]  # absolute values
    max_weight = max(raw_weights)
    importance = [w / max_weight for w in raw_weights]  # normalize to 0-1
    
    # Number of features
    N = len(features)
    
    # Angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Complete the values circle
    radar_values = importance + importance[:1]
    
    # Plot radar chart with enhanced styling
    ax.plot(angles, radar_values, 'o-', linewidth=4, color='#3498DB', 
            markersize=12, markerfacecolor='#3498DB', markeredgecolor='white', 
            markeredgewidth=2)
    ax.fill(angles, radar_values, alpha=0.3, color='#3498DB')
    
    # Set y-axis FIRST to establish the coordinate system
    ax.set_ylim(0, 1.5)  # Extra space for labels
    
    # Position labels close to their corresponding data points on the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Remove automatic labels to place manually
    
    # SIMPLE AND RELIABLE: Place labels at each angle position
    for i, (angle, label) in enumerate(zip(angles[:-1], features)):
        # Place label using polar coordinates directly
        # The radar chart expects (angle, radius) in polar coordinates
        label_radius = 1.2  # Outside the data area but within axes limits
        
        # Place the label at the exact angle with the specified radius
        # No conversion needed - matplotlib handles polar coordinates
        ax.text(angle, label_radius, label, 
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='#3498DB', alpha=0.95, linewidth=2))
    
    # Ensure we see all labels by setting proper limits
    ax.set_ylim(0, 1.5)  # Make sure this covers label_radius
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.2)
    
    # Enhanced title with more padding
    ax.set_title('Feature Importance in Entity Resolution\n(Normalized Weights)', 
                 fontweight='bold', fontsize=18, pad=40)
    
    # Remove the overlapping concentric circle labels - y-axis labels are sufficient
    # The grid lines and y-tick labels already provide the percentage information
    
    plt.tight_layout(pad=4.0)
    return fig

def main():
    """Generate and save the clean radar chart"""
    
    fig = create_clean_radar()
    
    output_path = Path(__file__).parent / "img" / "feature_radar_clean.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Clean feature radar saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()