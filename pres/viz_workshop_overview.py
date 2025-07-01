#!/usr/bin/env python3
"""
Workshop Overview Visualization
Shows learning objectives and journey map
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

def create_workshop_overview():
    """Create workshop overview and learning objectives"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.2, 1], width_ratios=[1, 1])
    
    fig.suptitle('From Words to Vectors: Yale AI Workshop Overview', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Learning objectives (top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    objectives = [
        {'title': 'Understand Text Embeddings', 'icon': '🧠', 'desc': 'How text becomes vectors that capture semantic meaning'},
        {'title': 'Apply to Entity Resolution', 'icon': '🔗', 'desc': 'Use embeddings to identify duplicate entities in catalogs'},
        {'title': 'Implement Classification', 'icon': '🤖', 'desc': 'Build classifiers with minimal labeled data'},
        {'title': 'Handle Production Scale', 'icon': '🚀', 'desc': 'Deploy systems processing millions of records'},
        {'title': 'Solve Real Problems', 'icon': '🎯', 'desc': 'Apply techniques to your own research challenges'}
    ]
    
    for i, obj in enumerate(objectives):
        x = 0.05 + i * 0.18
        
        # Objective box
        box = FancyBboxPatch(
            (x, 0.3), 0.16, 0.4,
            boxstyle="round,pad=0.02",
            facecolor='#3498DB',
            alpha=0.9,
            edgecolor='white',
            linewidth=2
        )
        ax1.add_patch(box)
        
        # Icon
        ax1.text(x + 0.08, 0.6, obj['icon'], ha='center', va='center', fontsize=20)
        
        # Title
        ax1.text(x + 0.08, 0.45, obj['title'], ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9)
        
        # Description
        ax1.text(x + 0.08, 0.15, obj['desc'], ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.text(0.5, 0.85, '🎯 LEARNING OBJECTIVES', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#3498DB')
    
    # 2. Workshop structure (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    structure = [
        {'phase': 'Part 1: Embeddings\nFundamentals', 'time': '15 min', 'color': '#E74C3C', 'y': 8},
        {'phase': 'Part 2: Domain\nClassification', 'time': '15 min', 'color': '#F39C12', 'y': 6},
        {'phase': 'Part 3: Complete\nPipeline', 'time': '18 min', 'color': '#27AE60', 'y': 4},
        {'phase': 'Discussion &\nQ&A', 'time': '7 min', 'color': '#9B59B6', 'y': 2}
    ]
    
    for part in structure:
        # Phase box
        box = FancyBboxPatch(
            (1, part['y']-0.6), 4, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=part['color'],
            alpha=0.9,
            edgecolor='white',
            linewidth=2
        )
        ax2.add_patch(box)
        
        # Phase text
        ax2.text(3, part['y'], part['phase'], ha='center', va='center', 
                fontweight='bold', color='white', fontsize=11)
        
        # Time
        ax2.text(5.5, part['y'], part['time'], ha='center', va='center', 
                fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor=part['color'], linewidth=2))
        
        # Arrow to next phase
        if part['y'] > 2:
            arrow = FancyArrowPatch(
                (3, part['y']-0.7), (3, part['y']-1.5),
                arrowstyle='->',
                color='#2C3E50',
                linewidth=2,
                alpha=0.8
            )
            ax2.add_patch(arrow)
    
    ax2.set_xlim(0, 7)
    ax2.set_ylim(1, 9.5)
    ax2.axis('off')
    ax2.set_title('Workshop Structure (45 minutes)', fontweight='bold', fontsize=12)
    
    # 3. Key concepts (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    concepts = [
        {'concept': 'Text Embeddings', 'example': 'text-embedding-3-small\n1536 dimensions', 'icon': '📊'},
        {'concept': 'Franz Schubert Problem', 'example': 'Composer vs Photographer\nSame name, different people', 'icon': '🎼'},
        {'concept': 'Domain Classification', 'example': 'Music vs Photography\nContextual disambiguation', 'icon': '🏷️'},
        {'concept': 'Vector Databases', 'example': 'Weaviate HNSW indexing\n99.23% efficiency gain', 'icon': '🗄️'},
        {'concept': 'Hot-deck Imputation', 'example': 'Semantic similarity\nfor missing data', 'icon': '🔄'},
        {'concept': 'Production ML', 'example': '99.55% precision\n17.6M records', 'icon': '🚀'}
    ]
    
    for i, concept in enumerate(concepts):
        y_pos = 8 - i * 1.3
        
        # Concept box
        box = FancyBboxPatch(
            (0.5, y_pos-0.4), 5, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#ECF0F1',
            edgecolor='#3498DB',
            linewidth=1
        )
        ax3.add_patch(box)
        
        # Icon
        ax3.text(1, y_pos, concept['icon'], ha='center', va='center', fontsize=16)
        
        # Concept name
        ax3.text(2, y_pos+0.15, concept['concept'], ha='left', va='center', 
                fontweight='bold', fontsize=10, color='#2C3E50')
        
        # Example
        ax3.text(2, y_pos-0.15, concept['example'], ha='left', va='center', 
                fontsize=8, style='italic', color='#7F8C8D')
    
    ax3.set_xlim(0, 6)
    ax3.set_ylim(1, 9)
    ax3.axis('off')
    ax3.set_title('Key Concepts & Examples', fontweight='bold', fontsize=12)
    
    # 4. Journey narrative (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Journey phases with detailed descriptions
    journey_phases = [
        {
            'title': '🔍 THE DISCOVERY',
            'content': 'We started with a simple question: How do we identify when two catalog records refer to the same person? The Franz Schubert case—composer vs photographer—became our driving example.',
            'color': '#E74C3C',
            'x': 0.02
        },
        {
            'title': '💡 THE INSIGHT',
            'content': 'Text embeddings provide semantic similarity, but similarity alone isn\'t enough. The threshold problem forced us to think beyond simple cosine distance.',
            'color': '#F39C12',
            'x': 0.21
        },
        {
            'title': '🎯 THE BREAKTHROUGH',
            'content': 'Domain classification provides the missing context. When combined with embeddings, we can distinguish between entities that pure similarity cannot separate.',
            'color': '#3498DB',
            'x': 0.40
        },
        {
            'title': '🚀 THE SOLUTION',
            'content': 'The complete pipeline combines embeddings, domain classification, vector databases, and feature engineering. Result: 99.55% precision at scale.',
            'color': '#27AE60',
            'x': 0.59
        },
        {
            'title': '📚 THE LEARNING',
            'content': 'This journey exemplifies real AI development: iterative, problem-driven, combining multiple techniques to solve challenges no single approach can handle.',
            'color': '#9B59B6',
            'x': 0.78
        }
    ]
    
    for phase in journey_phases:
        # Phase box
        box = FancyBboxPatch(
            (phase['x'], 0.1), 0.18, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=phase['color'],
            alpha=0.1,
            edgecolor=phase['color'],
            linewidth=2
        )
        ax4.add_patch(box)
        
        # Title
        ax4.text(phase['x'] + 0.09, 0.8, phase['title'], 
                ha='center', va='center', fontweight='bold', 
                fontsize=10, color=phase['color'])
        
        # Content
        ax4.text(phase['x'] + 0.09, 0.4, phase['content'], 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Add title for journey
    ax4.text(0.5, 0.95, '📖 THE STORY: From Problem to Production', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Add hands-on emphasis
    hands_on_text = (
        "🧪 HANDS-ON EXPERIENCE:\n"
        "• Interactive Jupyter notebooks\n"
        "• Real Yale catalog data\n"
        "• Live coding demonstrations\n"
        "• Franz Schubert case study\n"
        "• Production system walkthrough"
    )
    
    ax2.text(6.5, 6, hands_on_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Add technical stack
    tech_stack = (
        "🛠️ TECHNOLOGY STACK:\n"
        "• OpenAI text-embedding-3-small\n"
        "• Mistral Classifier Factory\n"
        "• Weaviate vector database\n"
        "• Python ML pipeline\n"
        "• Jupyter notebooks"
    )
    
    ax3.text(5.5, 3, tech_stack, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2))
    
    # Add audience note
    audience_text = (
        "👥 FOR ALL BACKGROUNDS:\n"
        "This workshop is designed for Yale graduate students\n"
        "from diverse fields—STEM to humanities.\n\n"
        "No AI experience required!\n"
        "We'll build understanding step by step."
    )
    
    ax1.text(0.5, 0.05, audience_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2),
            transform=ax1.transAxes)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the workshop overview visualization"""
    
    fig = create_workshop_overview()
    
    output_path = Path(__file__).parent / "img" / "workshop_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Workshop overview saved to {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()