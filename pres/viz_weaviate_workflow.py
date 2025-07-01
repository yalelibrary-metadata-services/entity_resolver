#!/usr/bin/env python3
"""
Weaviate Workflow Visualization
Shows vector database integration and workflow
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

def create_weaviate_workflow():
    """Create Weaviate workflow visualization"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    fig.suptitle('Weaviate Vector Database: Scaling Similarity Search', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Define workflow steps with positions
    steps = [
        {'name': 'Yale Catalog\nRecords', 'pos': (2, 8), 'color': '#3498DB', 'icon': '📚'},
        {'name': 'OpenAI\nEmbedding', 'pos': (5, 8), 'color': '#9B59B6', 'icon': '🧠'},
        {'name': 'Vector\nIndexing', 'pos': (8, 8), 'color': '#E67E22', 'icon': '🔗'},
        {'name': 'Weaviate\nStorage', 'pos': (11, 8), 'color': '#27AE60', 'icon': '🗄️'},
        {'name': 'Similarity\nSearch', 'pos': (14, 8), 'color': '#E74C3C', 'icon': '🔍'},
    ]
    
    # Draw workflow steps
    for i, step in enumerate(steps):
        x, y = step['pos']
        
        # Main step circle
        circle = Circle((x, y), 0.8, facecolor=step['color'], 
                       edgecolor='white', linewidth=3, alpha=0.9)
        ax.add_patch(circle)
        
        # Icon
        ax.text(x, y+0.2, step['icon'], ha='center', va='center', fontsize=20)
        
        # Step name
        ax.text(x, y-0.3, step['name'], ha='center', va='center', 
                fontweight='bold', fontsize=10, color=step['color'])
        
        # Step number
        ax.text(x, y+0.8, f'Step {i+1}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold', color='gray')
        
        # Arrow to next step
        if i < len(steps) - 1:
            next_x = steps[i+1]['pos'][0]
            arrow = FancyArrowPatch(
                (x+0.8, y), (next_x-0.8, y),
                arrowstyle='->',
                color='#2C3E50',
                linewidth=3,
                alpha=0.8
            )
            ax.add_patch(arrow)
    
    # Add detailed explanations below each step
    explanations = [
        "17.6M catalog records\nMixed languages\nVarying lengths\nRich metadata",
        "text-embedding-3-small\n1,536 dimensions\nBatch processing\n50% cost savings",
        "HNSW algorithm\nApproximate search\nOptimized for scale\nProduction tuning",
        "Vector database\nFast retrieval\nHorizontal scaling\nProduction ready",
        "Similarity queries\nNear-vector search\nCandidate filtering\n99.23% efficiency"
    ]
    
    for step, explanation in zip(steps, explanations):
        x, y = step['pos']
        ax.text(x, y-2, explanation, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=step['color'], linewidth=1, alpha=0.9))
    
    # Add performance metrics section
    metrics_text = (
        "📊 WEAVIATE PERFORMANCE METRICS:\n\n"
        "Scale Achievements:\n"
        "• 17.6M vectors indexed\n"
        "• Sub-second query response\n"
        "• 99.23% comparison reduction\n"
        "• Horizontal scalability\n\n"
        "Query Efficiency:\n"
        "• HNSW indexing algorithm\n"
        "• Approximate nearest neighbors\n"
        "• Configurable precision/speed tradeoff\n"
        "• Production-optimized parameters\n\n"
        "Integration Benefits:\n"
        "• Direct OpenAI API integration\n"
        "• Automatic embedding & indexing\n"
        "• RESTful and GraphQL APIs\n"
        "• Docker deployment ready"
    )
    
    ax.text(1, 5, metrics_text, ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Add comparison: traditional vs vector search
    ax.text(8, 4.5, 'TRADITIONAL APPROACH vs VECTOR DATABASE', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#F39C12', 
                     edgecolor='white', linewidth=2, alpha=0.9), color='white')
    
    # Traditional approach (left side)
    traditional_text = (
        "❌ TRADITIONAL PAIRWISE:\n\n"
        "Method: Compare every record\nto every other record\n\n"
        "Complexity: O(n²)\n"
        "17.6M records = 155 trillion comparisons\n\n"
        "Time: Computationally impossible\n"
        "Cost: Prohibitive\n"
        "Scalability: Does not scale"
    )
    
    ax.text(4, 2.5, traditional_text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FADBD8', 
                     edgecolor='#E74C3C', linewidth=2))
    
    # Vector approach (right side)
    vector_text = (
        "✅ VECTOR DATABASE:\n\n"
        "Method: Index vectors in\nhigh-dimensional space\n\n"
        "Complexity: O(log n)\n"
        "17.6M records = ~14K comparisons\n\n"
        "Time: Milliseconds per query\n"
        "Cost: Practical and affordable\n"
        "Scalability: Linear scaling"
    )
    
    ax.text(12, 2.5, vector_text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#D5F4E6', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Add arrow showing the improvement
    improvement_arrow = FancyArrowPatch(
        (6, 2.5), (10, 2.5),
        arrowstyle='->',
        color='#F39C12',
        linewidth=4,
        alpha=0.8
    )
    ax.add_patch(improvement_arrow)
    
    ax.text(8, 3, '99.23% REDUCTION', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#F39C12')
    
    # Add technical details sidebar
    tech_details = (
        "🛠️ TECHNICAL IMPLEMENTATION:\n\n"
        "Vector Indexing:\n"
        "• HNSW (Hierarchical NSW)\n"
        "• Configurable ef parameter\n"
        "• Max connections tuning\n"
        "• Environment-specific optimization\n\n"
        "Production Config:\n"
        "• Batch size: 1000 (prod) / 100 (dev)\n"
        "• EF: 256 (prod) / 128 (dev)\n"
        "• Connections: 128 (prod) / 64 (dev)\n"
        "• Pool size: 64 (prod) / 16 (dev)\n\n"
        "Deployment:\n"
        "• Docker Compose setup\n"
        "• Persistent storage\n"
        "• Health monitoring\n"
        "• Backup strategies"
    )
    
    ax.text(14.5, 5, tech_details, ha='left', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2))
    
    # Add query example
    query_example = (
        "🔍 EXAMPLE QUERY WORKFLOW:\n\n"
        "1. Input: 'Schubert, Franz - Symphony No. 9'\n"
        "2. Embed: Convert to 1,536-dim vector\n"
        "3. Search: Find nearest vectors in Weaviate\n"
        "4. Filter: Apply similarity threshold\n"
        "5. Return: Ranked candidate matches\n\n"
        "Query time: <100ms\n"
        "Candidates returned: ~50 records\n"
        "Accuracy: 99.55% precision"
    )
    
    ax.text(8, 1, query_example, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2))
    
    # Set axis limits and hide axes
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Add subtle grid for reference
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the Weaviate workflow visualization"""
    
    fig = create_weaviate_workflow()
    
    output_path = Path(__file__).parent / "img" / "weaviate_integration.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Weaviate workflow saved to {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()