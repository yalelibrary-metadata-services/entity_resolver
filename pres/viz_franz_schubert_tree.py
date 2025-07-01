#!/usr/bin/env python3
"""
Franz Schubert Decision Tree Visualization
Shows how the system disambiguates between composer and photographer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

def create_decision_box(ax, x, y, width, height, text, color, text_color='black'):
    """Create a decision box with text"""
    box = FancyBboxPatch(
        (x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=9, color=text_color, fontweight='bold',
            wrap=True)
    
    return box

def create_record_box(ax, x, y, title, person, domain, years, color):
    """Create a catalog record box"""
    box_height = 1.2
    box_width = 2.5
    
    # Main record box
    record_box = FancyBboxPatch(
        (x-box_width/2, y-box_height/2), box_width, box_height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(record_box)
    
    # Add record details
    ax.text(x, y+0.35, title[:35] + '...', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.text(x, y+0.1, f'Person: {person}', ha='center', va='center',
            fontsize=7, color='white')
    ax.text(x, y-0.1, f'Domain: {domain}', ha='center', va='center',
            fontsize=7, color='white', style='italic')
    ax.text(x, y-0.35, f'Years: {years}', ha='center', va='center',
            fontsize=7, color='white')

def create_arrow(ax, start_x, start_y, end_x, end_y, color='#2C3E50', style='->', width=1.5):
    """Create an arrow between points"""
    arrow = FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle=style,
        color=color,
        linewidth=width,
        alpha=0.8
    )
    ax.add_patch(arrow)

def create_franz_schubert_tree():
    """Create the Franz Schubert disambiguation decision tree"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Title
    fig.suptitle('Franz Schubert Disambiguation: From Problem to Solution', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Sample records (top level)
    ax.text(8, 10.5, '📚 YALE CATALOG RECORDS', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#3498DB', 
                     edgecolor='white', linewidth=2, alpha=0.9), color='white')
    
    # Record 1: Composer
    create_record_box(ax, 4, 9, 
                     'Symphony No. 9 in C major, D. 944',
                     'Schubert, Franz',
                     'Music, Sound, and Sonic Arts',
                     '1797-1828',
                     '#E74C3C')
    
    # Record 2: Photographer  
    create_record_box(ax, 12, 9,
                     'Archäologie und Photographie: fünfzig Beispiele',
                     'Schubert, Franz', 
                     'Documentary and Technical Arts',
                     '1930-1989',
                     '#E74C3C')
    
    # The question
    create_decision_box(ax, 8, 7.5, 4, 0.8,
                       '❓ SAME PERSON OR DIFFERENT?',
                       '#F39C12', 'white')
    
    # Draw arrows from records to question
    create_arrow(ax, 4, 8.4, 6.5, 7.8)
    create_arrow(ax, 12, 8.4, 9.5, 7.8)
    
    # Feature analysis section
    ax.text(8, 6.2, '🔍 FEATURE ANALYSIS', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#9B59B6', 
                     edgecolor='white', linewidth=1, alpha=0.9), color='white')
    
    # Feature boxes
    features = [
        ('Person Similarity\n0.98 (Very High)', 2, 5, '#27AE60', 'High name similarity\nsuggests same person'),
        ('Domain Classification\nMusic vs Documentation', 6, 5, '#E74C3C', 'Different domains\nsuggest different people'),
        ('Birth-Death Years\n1797-1828 vs 1930-1989', 10, 5, '#E74C3C', 'Different centuries\nconfirm different people'),
        ('Composite Similarity\n0.76 (Moderate)', 14, 5, '#F39C12', 'Similar but distinguishable\ncontent patterns')
    ]
    
    for feature_text, x, y, color, description in features:
        create_decision_box(ax, x, y, 2.2, 0.8, feature_text, color, 'white')
        ax.text(x, y-1, description, ha='center', va='center',
                fontsize=7, style='italic',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                         edgecolor=color, alpha=0.8))
        # Arrow from analysis to feature
        create_arrow(ax, 8, 5.8, x, y+0.4, width=1)
    
    # Decision logic
    create_decision_box(ax, 8, 3, 6, 1,
                       '⚖️ LOGISTIC REGRESSION CLASSIFIER\n'
                       'Combines all features with learned weights:\n'
                       'High person similarity (+) but different domains (-)\n'
                       'and different time periods (-) = DIFFERENT PEOPLE',
                       '#34495E', 'white')
    
    # Arrow to decision
    create_arrow(ax, 8, 5.6, 8, 3.5, width=2)
    
    # Final decision
    create_decision_box(ax, 8, 1, 8, 1,
                       '✅ FINAL DECISION: DIFFERENT ENTITIES\n'
                       'Franz Schubert (Composer, 1797-1828)\n'
                       'Franz Schubert (Photographer, 1930-1989)',
                       '#27AE60', 'white')
    
    # Arrow to final decision
    create_arrow(ax, 8, 2.5, 8, 1.5, width=2, color='#27AE60')
    
    # Add confidence scores
    ax.text(12, 1, 
            '📊 CONFIDENCE: 0.95\n'
            'System is 95% confident\n'
            'these are different people',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Add the key insight
    insight_text = (
        "🎯 KEY INSIGHT:\n"
        "While PERSON SIMILARITY suggests they might be the same,\n"
        "DOMAIN CLASSIFICATION and TEMPORAL FEATURES provide\n"
        "the decisive context to distinguish between them.\n\n"
        "This is why our system needed multiple features—\n"
        "no single signal was sufficient for disambiguation."
    )
    
    ax.text(1, 2.5, insight_text,
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2))
    
    # Add evolution note
    evolution_text = (
        "📈 EVOLUTION:\n"
        "Started with: Text similarity only\n"
        "Added: Domain classification\n"
        "Added: Temporal features\n"
        "Added: ML feature weighting\n"
        "Result: Robust disambiguation"
    )
    
    ax.text(14.5, 3.5, evolution_text,
            ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2))
    
    # Set axis limits and remove axes
    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Add very subtle background grid
    ax.grid(True, alpha=0.05)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the Franz Schubert decision tree"""
    
    # Create the visualization
    fig = create_franz_schubert_tree()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "franz_schubert_decision_tree.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Franz Schubert decision tree saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()