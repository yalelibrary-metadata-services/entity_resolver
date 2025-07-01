#!/usr/bin/env python3
"""
Pipeline Architecture Diagram
Shows the complete system architecture and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

def create_pipeline_architecture():
    """Create complete pipeline architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    fig.suptitle('Complete Entity Resolution Pipeline Architecture', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Define colors for different components
    colors = {
        'input': '#3498DB',
        'processing': '#E67E22',
        'ml': '#9B59B6',
        'storage': '#27AE60',
        'output': '#E74C3C'
    }
    
    # Component definitions with positions and properties
    components = [
        # Input Layer
        {'name': 'Yale Catalog\n17.6M Records', 'pos': (2, 10), 'size': (2, 1), 'color': colors['input'], 'type': 'input'},
        {'name': 'Ground Truth\nLabeled Pairs', 'pos': (2, 8.5), 'size': (2, 0.8), 'color': colors['input'], 'type': 'input'},
        
        # Preprocessing Layer
        {'name': 'Data Preprocessing\n& Deduplication', 'pos': (6, 10), 'size': (2.5, 1), 'color': colors['processing'], 'type': 'process'},
        {'name': 'Composite Field\nGeneration', 'pos': (6, 8.5), 'size': (2.5, 0.8), 'color': colors['processing'], 'type': 'process'},
        
        # Embedding Layer
        {'name': 'OpenAI API\ntext-embedding-3-small', 'pos': (10.5, 10), 'size': (3, 1), 'color': colors['ml'], 'type': 'ml'},
        {'name': 'Batch Processing\n(50% cost savings)', 'pos': (10.5, 8.5), 'size': (3, 0.8), 'color': colors['ml'], 'type': 'ml'},
        
        # Vector Database
        {'name': 'Weaviate\nVector Database', 'pos': (15, 9.25), 'size': (2.5, 1.5), 'color': colors['storage'], 'type': 'storage'},
        
        # Domain Classification
        {'name': 'Mistral Classifier\nDomain Taxonomy', 'pos': (6, 6.5), 'size': (2.5, 1), 'color': colors['ml'], 'type': 'ml'},
        {'name': 'Yale Taxonomy\n4 Domains, 20 Subfields', 'pos': (6, 5), 'size': (2.5, 0.8), 'color': colors['processing'], 'type': 'process'},
        
        # Hot-deck Imputation
        {'name': 'Vector Hot-deck\nImputation', 'pos': (10.5, 6.5), 'size': (3, 1), 'color': colors['ml'], 'type': 'ml'},
        {'name': 'Missing Subject\nEnhancement', 'pos': (10.5, 5), 'size': (3, 0.8), 'color': colors['processing'], 'type': 'process'},
        
        # Feature Engineering
        {'name': 'Feature Engineering\n5 Signals Combined', 'pos': (15, 6.5), 'size': (2.5, 1), 'color': colors['ml'], 'type': 'ml'},
        {'name': 'Person Similarity\nComposite Similarity\nDomain Classification\nTemporal Features\nInteraction Terms', 'pos': (15, 4.5), 'size': (2.5, 1.5), 'color': colors['processing'], 'type': 'process'},
        
        # ML Classification
        {'name': 'Logistic Regression\nClassifier', 'pos': (10.5, 3), 'size': (3, 1), 'color': colors['ml'], 'type': 'ml'},
        {'name': 'Training:\n14,930 pairs\n99.55% precision', 'pos': (6, 3), 'size': (2.5, 1), 'color': colors['processing'], 'type': 'process'},
        
        # Output Layer
        {'name': 'Entity Matches\nConfidence Scores', 'pos': (15, 2), 'size': (2.5, 1), 'color': colors['output'], 'type': 'output'},
        {'name': 'Entity Clusters\nTransitive Groups', 'pos': (10.5, 1), 'size': (3, 0.8), 'color': colors['output'], 'type': 'output'},
        {'name': 'Performance\nMetrics & Reports', 'pos': (6, 1), 'size': (2.5, 0.8), 'color': colors['output'], 'type': 'output'},
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        w, h = comp['size']
        
        # Create component box
        if comp['type'] == 'storage':
            # Special styling for database
            box = FancyBboxPatch(
                (x-w/2, y-h/2), w, h,
                boxstyle="round,pad=0.1",
                facecolor=comp['color'],
                edgecolor='white',
                linewidth=3,
                alpha=0.9
            )
        else:
            box = FancyBboxPatch(
                (x-w/2, y-h/2), w, h,
                boxstyle="round,pad=0.05",
                facecolor=comp['color'],
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
        ax.add_patch(box)
        
        # Add component text
        ax.text(x, y, comp['name'], ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.3))
    
    # Define data flow arrows
    flows = [
        # Input to preprocessing
        ((3, 10), (5.25, 10)),
        ((3, 8.5), (5.25, 8.5)),
        
        # Preprocessing to embedding
        ((8.25, 10), (9.25, 10)),
        ((8.25, 8.5), (9.25, 8.5)),
        
        # Embedding to vector DB
        ((13.5, 9.25), (13.75, 9.25)),
        
        # To domain classification
        ((7.25, 9), (7.25, 7.5)),
        ((7.25, 5.8), (7.25, 5.4)),
        
        # To hot-deck imputation
        ((12, 9), (12, 7.5)),
        ((12, 5.8), (12, 5.4)),
        
        # To feature engineering
        ((15, 8.5), (15, 7.5)),
        ((16.25, 6), (16.25, 5.25)),
        
        # Domain classification to features
        ((8.25, 6.5), (13.75, 6.5)),
        
        # Hot-deck to features
        ((13.5, 6.5), (13.75, 6.5)),
        
        # Features to ML
        ((15, 5.75), (12, 4)),
        
        # Training data to ML
        ((8.25, 3), (9.25, 3)),
        
        # ML to outputs
        ((12, 3.5), (15, 2.5)),
        ((12, 2.5), (12, 1.4)),
        ((10, 3), (7.25, 1.4)),
    ]
    
    # Draw arrows
    for start, end in flows:
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->',
            color='#2C3E50',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(arrow)
    
    # Add layer labels
    layer_labels = [
        ('INPUT DATA', 1, 11, colors['input']),
        ('PREPROCESSING', 5, 11, colors['processing']),
        ('EMBEDDING & STORAGE', 12.5, 11, colors['ml']),
        ('FEATURE ENGINEERING', 12.5, 7.5, colors['processing']),
        ('MACHINE LEARNING', 8.5, 4, colors['ml']),
        ('OUTPUTS & RESULTS', 12.5, 1.5, colors['output'])
    ]
    
    for label, x, y, color in layer_labels:
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                         edgecolor='white', linewidth=2, alpha=0.2))
    
    # Add performance metrics box
    metrics_text = (
        "🎯 SYSTEM PERFORMANCE:\n"
        "• Precision: 99.55%\n"
        "• Recall: 82.48%\n"
        "• F1-Score: 90.22%\n"
        "• Scale: 17.6M records\n"
        "• Efficiency: 99.23% reduction\n"
        "• Processing: Real-time capability"
    )
    
    ax.text(1, 6, metrics_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=3))
    
    # Add technology stack
    tech_stack = (
        "🛠️ TECHNOLOGY STACK:\n"
        "• OpenAI text-embedding-3-small\n"
        "• Weaviate vector database\n"
        "• Mistral Classifier Factory\n"
        "• Custom ML pipeline (Python)\n"
        "• Docker containerization\n"
        "• Production monitoring"
    )
    
    ax.text(1, 3.5, tech_stack, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=3))
    
    # Add data flow legend
    legend_elements = [
        ('📊 Data Flow', '#2C3E50', '→'),
        ('🔵 Input/Output', colors['input'], '●'),
        ('🟠 Processing', colors['processing'], '●'),
        ('🟣 Machine Learning', colors['ml'], '●'),
        ('🟢 Storage', colors['storage'], '●'),
        ('🔴 Results', colors['output'], '●')
    ]
    
    legend_y = 0.5
    for i, (label, color, symbol) in enumerate(legend_elements):
        ax.text(18.5, legend_y - i*0.3, f'{symbol} {label}', 
                ha='left', va='center', fontsize=9, color=color, fontweight='bold')
    
    # Add title for Franz Schubert example
    franz_example = (
        "🎼 FRANZ SCHUBERT EXAMPLE:\n\n"
        "Input: Two 'Schubert, Franz' records\n"
        "↓ Embedding: High text similarity (0.76)\n" 
        "↓ Domain: Music vs Photography\n"
        "↓ Features: Conflicting signals\n"
        "↓ ML: Domain difference decisive\n"
        "Output: Correctly identified as different people"
    )
    
    ax.text(18.5, 8, franz_example, ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2))
    
    # Set axis properties
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Add subtle background grid
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the pipeline architecture diagram"""
    
    fig = create_pipeline_architecture()
    
    output_path = Path(__file__).parent / "img" / "pipeline_architecture.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Pipeline architecture saved to {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()