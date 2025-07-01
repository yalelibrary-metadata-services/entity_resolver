#!/usr/bin/env python3
"""
Feature Descriptions Cards
Clean visual explanations of each feature
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

def create_feature_descriptions():
    """Create clean feature description cards"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    
    # Feature data with real Yale weights and enhanced descriptions
    features_data = [
        {
            'name': 'Person Similarity',
            'weight': '+0.603',
            'icon': '👤',
            'description': 'Cosine similarity between\nperson name embeddings\n\n"Schubert, Franz" vs "Schubert, Franz"\n→ High similarity (0.98)',
            'interpretation': 'Same person names should\nhave very high similarity',
            'color': '#3498DB'
        },
        {
            'name': 'Full Record Similarity',
            'weight': '+1.458',
            'icon': '📄',
            'description': 'Cosine similarity between\ncomplete record embeddings\n\nTitle + Person + Subjects + ...\n→ Holistic content matching',
            'interpretation': 'Same people should have\nsimilar associated works',
            'color': '#9B59B6'
        },
        {
            'name': 'Person-Title Interaction',
            'weight': '+1.017',
            'icon': '⚡',
            'description': 'Squared product of person\nand title similarities\n\nCaptures when BOTH signals\nare strong simultaneously',
            'interpretation': 'Strong interaction suggests\nspecific person-work pairing',
            'color': '#F39C12'
        },
        {
            'name': 'Birth-Death Match',
            'weight': '+2.514',
            'icon': '📅',
            'description': 'Temporal consistency check\nwith 2-year tolerance\n\nHandles missing/approximate\ndates gracefully',
            'interpretation': 'Same people should have\nconsistent lifespans',
            'color': '#27AE60'
        },
        {
            'name': 'Domain Difference',
            'weight': '-1.812',
            'icon': '🎯',
            'description': 'Different activity domains\nsuggest different people\n\nMusic vs Photography\n→ Strong disambiguation signal',
            'interpretation': 'Different domains strongly\nindicate different people',
            'color': '#E74C3C'
        }
    ]
    
    # Layout parameters with better spacing
    cards_per_row = 3
    card_width = 0.28
    card_height = 0.42
    margin_x = 0.08
    margin_y = 0.12
    
    for i, feature in enumerate(features_data):
        # Calculate position
        row = i // cards_per_row
        col = i % cards_per_row
        
        x = margin_x + col * (card_width + margin_x)
        y = 0.9 - row * (card_height + margin_y)
        
        # Create feature card
        card = FancyBboxPatch(
            (x, y - card_height), card_width, card_height,
            boxstyle="round,pad=0.02",
            facecolor=feature['color'],
            alpha=0.1,
            edgecolor=feature['color'],
            linewidth=3
        )
        ax.add_patch(card)
        
        # Feature icon and name
        ax.text(x + card_width/2, y - 0.05, feature['icon'], 
                ha='center', va='center', fontsize=24)
        
        ax.text(x + card_width/2, y - 0.12, feature['name'], 
                ha='center', va='center', fontsize=14, fontweight='bold', 
                color=feature['color'])
        
        # Weight with visual emphasis
        weight_color = '#27AE60' if '+' in feature['weight'] else '#E74C3C'
        ax.text(x + card_width/2, y - 0.18, f"Weight: {feature['weight']}", 
                ha='center', va='center', fontsize=12, fontweight='bold', 
                color=weight_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor=weight_color, linewidth=2))
        
        # Description
        ax.text(x + card_width/2, y - 0.28, feature['description'], 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
        
        # Interpretation
        ax.text(x + card_width/2, y - 0.36, feature['interpretation'], 
                ha='center', va='center', fontsize=9, style='italic',
                color=feature['color'], fontweight='bold')
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.98, 'Feature Engineering: Understanding the Signals', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add footer explanation
    footer_text = (
        "💡 FEATURE ENGINEERING: Each feature captures a different aspect of entity similarity. "
        "The model learns optimal weights to combine these signals for accurate entity resolution."
    )
    
    ax.text(0.5, 0.02, footer_text, 
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9FA', 
                     edgecolor='#2C3E50', linewidth=2))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the feature descriptions"""
    
    fig = create_feature_descriptions()
    
    output_path = Path(__file__).parent / "img" / "feature_descriptions.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Feature descriptions saved to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()