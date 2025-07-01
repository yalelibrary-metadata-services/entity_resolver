#!/usr/bin/env python3
"""
Embedding Evolution Timeline
Shows progression from Word2Vec to modern transformers
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_embedding_evolution():
    """Create embedding evolution timeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    fig.suptitle('Evolution of Text Embeddings: From Word2Vec to Production AI', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Timeline data
    timeline_data = [
        {
            'year': 2013,
            'model': 'Word2Vec',
            'innovation': 'Neural word vectors',
            'dims': 300,
            'context': 'No',
            'limitation': 'Word-level only',
            'color': '#95A5A6',
            'x': 1
        },
        {
            'year': 2014,
            'model': 'GloVe',
            'innovation': 'Global matrix factorization',
            'dims': 300,
            'context': 'No',
            'limitation': 'Static embeddings',
            'color': '#3498DB',
            'x': 3
        },
        {
            'year': 2016,
            'model': 'FastText',
            'innovation': 'Subword information',
            'dims': 300,
            'context': 'Limited',
            'limitation': 'Still word-level',
            'color': '#9B59B6',
            'x': 5
        },
        {
            'year': 2018,
            'model': 'BERT',
            'innovation': 'Bidirectional context',
            'dims': 768,
            'context': 'Yes',
            'limitation': '512 token limit',
            'color': '#E67E22',
            'x': 7
        },
        {
            'year': 2020,
            'model': 'GPT-3',
            'innovation': 'Large-scale autoregressive',
            'dims': 12288,
            'context': 'Yes',
            'limitation': 'Expensive, complex',
            'color': '#E74C3C',
            'x': 9
        },
        {
            'year': 2022,
            'model': 'OpenAI Ada',
            'innovation': 'Dedicated embedding model',
            'dims': 1536,
            'context': 'Yes',
            'limitation': 'API dependency',
            'color': '#F39C12',
            'x': 11
        },
        {
            'year': 2024,
            'model': 'text-embedding-3',
            'innovation': 'Improved efficiency & performance',
            'dims': 1536,
            'context': 'Yes',
            'limitation': 'Our choice ✅',
            'color': '#27AE60',
            'x': 13
        }
    ]
    
    # Draw timeline backbone
    ax.plot([0.5, 13.5], [5, 5], 'k-', linewidth=3, alpha=0.3)
    
    # Draw model progression
    for i, model in enumerate(timeline_data):
        x = model['x']
        
        # Main model box
        box = FancyBboxPatch(
            (x-0.8, 4.3), 1.6, 1.4,
            boxstyle="round,pad=0.1",
            facecolor=model['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Model name and year
        ax.text(x, 5.4, model['model'], ha='center', va='center', 
                fontweight='bold', color='white', fontsize=11)
        ax.text(x, 4.8, str(model['year']), ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9)
        ax.text(x, 4.4, f"{model['dims']}D", ha='center', va='center', 
                fontweight='bold', color='white', fontsize=8)
        
        # Timeline point
        ax.plot(x, 5, 'o', markersize=10, color='white', 
                markeredgecolor=model['color'], markeredgewidth=3)
        
        # Innovation (above)
        ax.text(x, 6.5, '💡 ' + model['innovation'], ha='center', va='center', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor=model['color'], linewidth=1, alpha=0.9))
        
        # Context awareness
        context_color = '#27AE60' if model['context'] == 'Yes' else '#E74C3C'
        ax.text(x, 3.5, f"Context: {model['context']}", ha='center', va='center', 
                fontsize=8, fontweight='bold', color=context_color)
        
        # Limitation (below)
        limit_color = '#27AE60' if 'choice' in model['limitation'] else '#E74C3C'
        ax.text(x, 2.8, model['limitation'], ha='center', va='center', 
                fontsize=8, style='italic', color=limit_color,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                         edgecolor=limit_color, alpha=0.8))
        
        # Arrow to next model
        if i < len(timeline_data) - 1:
            arrow = FancyArrowPatch(
                (x+0.8, 5), (timeline_data[i+1]['x']-0.8, 5),
                arrowstyle='->',
                color='#2C3E50',
                linewidth=2,
                alpha=0.6
            )
            ax.add_patch(arrow)
    
    # Add capability comparison chart
    capabilities = ['Word-level', 'Sentence-level', 'Document-level', 'Context-aware', 'Production-ready']
    models_subset = ['Word2Vec', 'BERT', 'GPT-3', 'text-embedding-3']
    
    # Capability matrix (simplified)
    cap_matrix = {
        'Word2Vec': [1, 0, 0, 0, 0.5],
        'BERT': [1, 1, 0.5, 1, 0.7],
        'GPT-3': [1, 1, 1, 1, 0.6],
        'text-embedding-3': [1, 1, 1, 1, 1]
    }
    
    # Draw capability heatmap
    ax.text(7, 1.5, 'Capability Evolution', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    for i, capability in enumerate(capabilities):
        # Capability label
        ax.text(2.5, 1 - i*0.15, capability, ha='right', va='center', 
                fontsize=9, fontweight='bold')
        
        for j, model in enumerate(models_subset):
            x_pos = 3.5 + j * 1.5
            y_pos = 1 - i*0.15
            
            # Capability level
            level = cap_matrix[model][i]
            colors = {0: '#E74C3C', 0.5: '#F39C12', 0.6: '#F39C12', 0.7: '#3498DB', 1: '#27AE60'}
            color = colors.get(level, '#95A5A6')
            
            # Draw capability indicator
            circle = plt.Circle((x_pos, y_pos), 0.05, facecolor=color, 
                              edgecolor='white', linewidth=1)
            ax.add_patch(circle)
    
    # Model labels for capability matrix
    for j, model in enumerate(models_subset):
        x_pos = 3.5 + j * 1.5
        ax.text(x_pos, 1.3, model, ha='center', va='center', 
                fontsize=9, fontweight='bold', rotation=45)
    
    # Add legend for capability matrix
    legend_items = [
        ('Full capability', '#27AE60'),
        ('Partial capability', '#3498DB'),
        ('Limited capability', '#F39C12'),
        ('No capability', '#E74C3C')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = 0.1 - i*0.1
        circle = plt.Circle((10, y_pos), 0.03, facecolor=color, 
                          edgecolor='white', linewidth=1)
        ax.add_patch(circle)
        ax.text(10.2, y_pos, label, ha='left', va='center', fontsize=8)
    
    # Add Yale's decision rationale
    decision_text = (
        "🎯 WHY text-embedding-3-small FOR YALE:\n\n"
        "✅ Optimized for similarity tasks\n"
        "✅ 1,536 dimensions (sweet spot)\n"
        "✅ Cost-effective API pricing\n"
        "✅ Reliable production service\n"
        "✅ Excellent multilingual support\n"
        "✅ Fast inference speed\n"
        "✅ Consistent quality\n\n"
        "📊 Production Performance:\n"
        "• 99.55% precision\n"
        "• 17.6M records processed\n"
        "• <100ms embedding time\n"
        "• $26K total embedding cost"
    )
    
    ax.text(13.5, 7, decision_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Add technical progression insights
    progression_text = (
        "📈 KEY TECHNICAL PROGRESSIONS:\n\n"
        "🧠 Representation Power:\n"
        "Word → Subword → Sentence → Document\n\n"
        "🎯 Context Understanding:\n"
        "Static → Dynamic → Bidirectional → Multi-scale\n\n"
        "⚡ Efficiency Evolution:\n"
        "Research → Engineering → Production → Optimization\n\n"
        "🌍 Application Scope:\n"
        "Single language → Multilingual → Cross-modal → Domain-specific"
    )
    
    ax.text(0.5, 7, progression_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2))
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    
    # Add subtle grid
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the embedding evolution visualization"""
    
    fig = create_embedding_evolution()
    
    output_path = Path(__file__).parent / "img" / "embedding_evolution_chart.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Embedding evolution saved to {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()