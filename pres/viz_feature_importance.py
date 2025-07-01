#!/usr/bin/env python3
"""
Feature Importance Radar Chart
Shows ML feature weights and their interpretations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

def create_feature_importance():
    """Create feature importance radar chart and analysis"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], width_ratios=[1, 1])
    
    # Main title
    fig.suptitle('Feature Engineering: Multi-Signal Entity Resolution', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Feature data (based on actual Yale system)
    features = [
        'Person\nCosine',
        'Composite\nCosine', 
        'Person-Title\nSquared',
        'Domain\nDissimilarity',
        'Birth-Death\nMatch'
    ]
    
    # Actual weights from logistic regression (normalized for visualization)
    weights = [0.85, 0.72, 0.45, -0.92, 0.38]  # Domain dissimilarity is negative
    
    # Feature descriptions
    descriptions = [
        'Similarity between\nperson name embeddings\n(Positive weight)',
        'Similarity between\nfull record embeddings\n(Positive weight)',
        'Interaction term:\nperson × title similarity\n(Positive weight)',
        'Different domains suggest\ndifferent people\n(Negative weight)',
        'Birth/death year matching\nwith tolerance\n(Positive weight)'
    ]
    
    # Feature importance (absolute values for radar)
    importance = [abs(w) for w in weights]
    
    # 1. Radar chart (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    # Number of features
    N = len(features)
    
    # Angles for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Values for radar (normalize to 0-1 scale)
    max_importance = max(importance)
    radar_values = [imp / max_importance for imp in importance]
    radar_values += radar_values[:1]  # Complete the circle
    
    # Plot radar chart
    ax1.plot(angles, radar_values, 'o-', linewidth=2, color='#3498DB', markersize=8)
    ax1.fill(angles, radar_values, alpha=0.25, color='#3498DB')
    
    # Add feature labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(features, fontsize=10, fontweight='bold')
    
    # Set y-axis limits and labels
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax1.set_title('Feature Importance\n(Normalized Absolute Weights)', 
                 fontweight='bold', pad=20, y=1.08)
    
    # 2. Weight interpretation (top right)  
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create horizontal bar chart
    y_positions = range(len(features))
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    bars = ax2.barh(y_positions, weights, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        width = bar.get_width()
        ax2.text(width + (0.05 if width > 0 else -0.05), bar.get_y() + bar.get_height()/2,
                f'{weight:.2f}',
                ha='left' if width > 0 else 'right', va='center', fontweight='bold')
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(features, fontsize=10)
    ax2.set_xlabel('Logistic Regression Weight', fontweight='bold')
    ax2.set_title('Feature Weights with Direction\n(Positive vs Negative Impact)', 
                 fontweight='bold', pad=20)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend for weight interpretation
    ax2.text(0.6, 4.2, 'Positive Weight\nIncreases match probability', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
    
    ax2.text(-0.6, 3.2, 'Negative Weight\nDecreases match probability', 
            ha='center', va='center', fontsize=9, fontweight='bold', color='red',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
    
    # 3. Feature descriptions (bottom)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create feature description boxes
    box_width = 0.18
    box_height = 0.8
    
    for i, (feature, desc, weight) in enumerate(zip(features, descriptions, weights)):
        x = 0.02 + i * 0.196
        
        # Color based on weight direction
        color = '#27AE60' if weight > 0 else '#E74C3C'
        
        # Create feature box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (x, 0.1), box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=0.1,
            edgecolor=color,
            linewidth=2
        )
        ax3.add_patch(box)
        
        # Feature name
        ax3.text(x + box_width/2, 0.8, feature.replace('\n', ' '), 
                ha='center', va='center', fontweight='bold', 
                fontsize=11, color=color)
        
        # Weight value
        ax3.text(x + box_width/2, 0.65, f'Weight: {weight:.2f}', 
                ha='center', va='center', fontweight='bold', 
                fontsize=10, color='black')
        
        # Description
        ax3.text(x + box_width/2, 0.35, desc, 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.text(0.5, 0.05, 'Feature Interpretations and Impact on Entity Resolution Decisions', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add Franz Schubert example analysis
    franz_analysis = (
        "🎼 FRANZ SCHUBERT EXAMPLE ANALYSIS:\n\n"
        "Composer vs Photographer decision:\n"
        "• Person Cosine: 0.98 (HIGH) → +0.83 contribution\n"
        "• Composite Cosine: 0.76 (MED) → +0.55 contribution\n"
        "• Domain Dissimilarity: 1.0 (DIFF) → -0.92 contribution\n"
        "• Birth-Death Match: 0.0 (NO) → +0.0 contribution\n\n"
        "NET EFFECT: Positive similarity signals are\nOVERRULED by domain difference\n"
        "DECISION: Different people (correct!)"
    )
    
    # Add this as a side panel
    ax2.text(1.2, 2.5, franz_analysis, 
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2),
            transform=ax2.transData)
    
    # Add model performance summary
    performance_text = (
        "🎯 MODEL PERFORMANCE:\n"
        "• Precision: 99.55%\n"
        "• Recall: 82.48%\n"
        "• F1-Score: 90.22%\n"
        "• Training: Logistic Regression\n"
        "• Features: 5 engineered signals\n"
        "• Scale: 17.6M records"
    )
    
    ax1.text(1.3, 0.5, performance_text, 
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2),
            transform=ax1.transAxes)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the feature importance visualization"""
    
    # Create the visualization
    fig = create_feature_importance()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "feature_importance_radar.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Feature importance radar saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()