#!/usr/bin/env python3
"""
Franz Schubert Feature Analysis
Detailed breakdown of how features combine for the Schubert decision
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_schubert_analysis():
    """Create detailed Schubert feature analysis visualization"""
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.6, 1.4, 1.0], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.1)
    
    fig.suptitle('🎼 Franz Schubert Case Study: Feature Analysis in Action', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Record comparison (top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    # Record A (Composer)
    record_a = FancyBboxPatch(
        (0.05, 0.2), 0.4, 0.6,
        boxstyle="round,pad=0.03",
        facecolor='#E8F4F8',
        edgecolor='#3498DB',
        linewidth=3
    )
    ax1.add_patch(record_a)
    
    ax1.text(0.25, 0.7, '🎵 RECORD A: Composer', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#3498DB')
    ax1.text(0.25, 0.5, 'Person: Schubert, Franz\nTitle: Symphony No. 9 in C major\nSubjects: Classical music; Orchestral works\nGenre: Symphonic music', 
            ha='center', va='center', fontsize=11)
    
    # Record B (Photographer)
    record_b = FancyBboxPatch(
        (0.55, 0.2), 0.4, 0.6,
        boxstyle="round,pad=0.03",
        facecolor='#FADBD8',
        edgecolor='#E74C3C',
        linewidth=3
    )
    ax1.add_patch(record_b)
    
    ax1.text(0.75, 0.7, '📸 RECORD B: Photographer', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#E74C3C')
    ax1.text(0.75, 0.5, 'Person: Schubert, Franz\nTitle: Archäologie und Photographie\nSubjects: Photography in archaeology\nGenre: Documentary photography', 
            ha='center', va='center', fontsize=11)
    
    # VS indicator
    ax1.text(0.5, 0.5, 'VS', ha='center', va='center', 
            fontsize=20, fontweight='bold', color='#F39C12',
            bbox=dict(boxstyle="circle,pad=0.3", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2))
    
    # 2. Feature calculations (middle)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    
    # Feature analysis data (using real Yale weights and realistic values)
    features_analysis = [
        {
            'name': 'Person Similarity',
            'value': 0.98,
            'weight': 0.603,
            'contribution': 0.59,
            'explanation': 'Identical person names\n→ Nearly perfect similarity',
            'color': '#3498DB',
            'x': 0.1
        },
        {
            'name': 'Full Record Similarity',
            'value': 0.52,
            'weight': 1.458,
            'contribution': 0.76,
            'explanation': 'Some shared context\nbut different subjects',
            'color': '#9B59B6',
            'x': 0.3
        },
        {
            'name': 'Person-Title Interaction',
            'value': 0.18,
            'weight': 1.017,
            'contribution': 0.18,
            'explanation': 'Low interaction\n(different domains)',
            'color': '#F39C12',
            'x': 0.5
        },
        {
            'name': 'Domain Difference',
            'value': 1.0,
            'weight': -1.812,
            'contribution': -1.81,
            'explanation': 'Completely different domains\n→ Strong negative signal',
            'color': '#E74C3C',
            'x': 0.7
        },
        {
            'name': 'Birth-Death Match',
            'value': 0.0,
            'weight': 2.514,
            'contribution': 0.0,
            'explanation': 'No birth/death info\navailable for comparison',
            'color': '#27AE60',
            'x': 0.9
        }
    ]
    
    # Draw feature analysis
    for feature in features_analysis:
        x = feature['x']
        
        # Feature box
        box_color = feature['color']
        contribution_color = '#27AE60' if feature['contribution'] > 0 else '#E74C3C'
        
        # Main feature box with better spacing
        feature_box = FancyBboxPatch(
            (x-0.09, 0.45), 0.18, 0.4,
            boxstyle="round,pad=0.02",
            facecolor=box_color,
            alpha=0.2,
            edgecolor=box_color,
            linewidth=2
        )
        ax2.add_patch(feature_box)
        
        # Feature name with better positioning
        ax2.text(x, 0.82, feature['name'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color=box_color)
        
        # Value and weight with improved spacing
        ax2.text(x, 0.74, f'Value: {feature["value"]:.2f}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        ax2.text(x, 0.69, f'Weight: {feature["weight"]:.2f}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Contribution (highlighted) with better positioning
        ax2.text(x, 0.62, f'= {feature["contribution"]:.2f}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color=contribution_color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                         edgecolor=contribution_color, linewidth=2))
        
        # Explanation with better spacing
        ax2.text(x, 0.38, feature['explanation'], ha='center', va='center', 
                fontsize=9, style='italic')
        
        # Arrow to final calculation
        if x < 0.85:
            arrow = FancyArrowPatch(
                (x+0.08, 0.55), (x+0.12, 0.55),
                arrowstyle='->',
                color='#2C3E50',
                linewidth=2,
                alpha=0.6
            )
            ax2.add_patch(arrow)
    
    # 3. Final calculation (bottom)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    
    # Calculation summary (using real Yale weights)
    positive_sum = 0.59 + 0.76 + 0.18 + 0.0
    negative_sum = -1.81
    net_score = positive_sum + negative_sum
    
    # Calculation boxes
    calc_text = (
        f"📊 FINAL CALCULATION:\n\n"
        f"Positive contributions: {positive_sum:.2f}\n"
        f"Negative contributions: {negative_sum:.2f}\n"
        f"Net score: {net_score:.2f}\n\n"
        f"Decision threshold: 0.65\n"
        f"Result: {net_score:.2f} < 0.65 → DIFFERENT PEOPLE ✅"
    )
    
    ax3.text(0.3, 0.5, calc_text, ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#F8F9FA', 
                     edgecolor='#2C3E50', linewidth=3))
    
    # Key insight
    insight_text = (
        "🔍 KEY INSIGHT:\n\n"
        "Even though the person names are identical (0.98 similarity), "
        "the DOMAIN DIFFERENCE feature provides the decisive signal.\n\n"
        "This demonstrates why multi-feature approaches are essential "
        "for complex entity resolution tasks.\n\n"
        "The model learned that domain differences are highly predictive "
        "of different people, even when names match perfectly."
    )
    
    ax3.text(0.7, 0.5, insight_text, ha='center', va='center', 
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Visual flow arrows
    ax2.text(0.5, 0.15, '↓ COMBINE ALL FEATURES ↓', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#F39C12')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the Schubert feature analysis"""
    
    fig = create_schubert_analysis()
    
    output_path = Path(__file__).parent / "img" / "schubert_feature_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Schubert feature analysis saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()