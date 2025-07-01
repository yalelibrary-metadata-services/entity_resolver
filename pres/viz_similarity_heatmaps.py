#!/usr/bin/env python3
"""
Similarity Heatmaps Visualization
Shows vector similarity patterns and clustering
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

def create_similarity_heatmaps():
    """Create similarity heatmap visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], width_ratios=[1, 1])
    
    # Main title
    fig.suptitle('Semantic Similarity Analysis: Vector Embeddings in Action', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Sample entities for demonstration
    entities = [
        'Schubert, Franz (Composer)',
        'Schubert, Franz (Photographer)', 
        'Bach, Johann Sebastian',
        'García Márquez, Gabriel',
        'Einstein, Albert',
        'Adams, Ansel (Photographer)',
        'Mozart, Wolfgang Amadeus',
        'Neruda, Pablo'
    ]
    
    # Simulated similarity matrix (based on realistic patterns)
    similarity_matrix = np.array([
        [1.00, 0.76, 0.42, 0.18, 0.15, 0.31, 0.45, 0.22],  # Schubert Composer
        [0.76, 1.00, 0.19, 0.16, 0.13, 0.68, 0.21, 0.18],  # Schubert Photographer  
        [0.42, 0.19, 1.00, 0.24, 0.17, 0.28, 0.89, 0.26],  # Bach
        [0.18, 0.16, 0.24, 1.00, 0.22, 0.19, 0.21, 0.85],  # García Márquez
        [0.15, 0.13, 0.17, 0.22, 1.00, 0.16, 0.18, 0.24],  # Einstein
        [0.31, 0.68, 0.28, 0.19, 0.16, 1.00, 0.25, 0.21],  # Adams
        [0.45, 0.21, 0.89, 0.21, 0.18, 0.25, 1.00, 0.23],  # Mozart
        [0.22, 0.18, 0.26, 0.85, 0.24, 0.21, 0.23, 1.00]   # Neruda
    ])
    
    # 1. Main similarity heatmap
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create heatmap
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    heatmap = sns.heatmap(similarity_matrix, 
                         mask=mask,
                         annot=True, 
                         fmt='.2f',
                         cmap='RdYlBu_r',
                         center=0.5,
                         vmin=0, vmax=1,
                         square=True,
                         xticklabels=entities,
                         yticklabels=entities,
                         cbar_kws={'label': 'Cosine Similarity'},
                         ax=ax1)
    
    ax1.set_title('Entity Similarity Matrix\n(Lower triangle shows cosine similarity between embeddings)', 
                 fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Add highlighting boxes for key relationships
    # Franz Schubert relationship
    rect1 = Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', linewidth=3)
    ax1.add_patch(rect1)
    ax1.text(0.5, 0.5, 'Franz Schubert\nProblem!', ha='center', va='center',
             fontweight='bold', color='red', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    # Music cluster
    rect2 = Rectangle((0, 2), 1, 1, fill=False, edgecolor='blue', linewidth=2)
    ax1.add_patch(rect2)
    rect3 = Rectangle((6, 2), 1, 1, fill=False, edgecolor='blue', linewidth=2)
    ax1.add_patch(rect3)
    
    # Literature cluster  
    rect4 = Rectangle((3, 7), 1, 1, fill=False, edgecolor='green', linewidth=2)
    ax1.add_patch(rect4)
    
    # Photography cluster
    rect5 = Rectangle((1, 5), 1, 1, fill=False, edgecolor='purple', linewidth=2)
    ax1.add_patch(rect5)
    
    # 2. Threshold analysis
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Extract all similarity values (excluding diagonal)
    similarities = []
    labels = []
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            similarities.append(similarity_matrix[i, j])
            if i == 0 and j == 1:  # Franz Schubert pair
                labels.append('Franz Schubert\n(Different people)')
            elif (i in [0, 2, 6] and j in [0, 2, 6]) or (i in [3, 7] and j in [3, 7]):
                labels.append('Same domain')
            else:
                labels.append('Different domain')
    
    # Create scatter plot
    colors = ['red' if 'Franz' in label else 'blue' if 'Same' in label else 'gray' 
              for label in labels]
    
    y_positions = np.random.normal(0, 0.1, len(similarities))  # Add jitter
    scatter = ax2.scatter(similarities, y_positions, c=colors, alpha=0.7, s=60)
    
    # Add threshold lines
    thresholds = [0.5, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        ax2.axvline(x=threshold, color='orange', linestyle='--', alpha=0.6)
        ax2.text(threshold, 0.4, f'{threshold}', rotation=90, 
                ha='right', va='bottom', color='orange', fontweight='bold')
    
    # Highlight Franz Schubert
    franz_similarity = similarity_matrix[0, 1]
    ax2.annotate('Franz Schubert Problem\n(0.76 similarity)', 
                xy=(franz_similarity, 0), xytext=(0.6, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='red', alpha=0.9))
    
    ax2.set_xlabel('Cosine Similarity', fontweight='bold')
    ax2.set_title('Threshold Problem Visualization\n(No single threshold works for all cases)', 
                 fontweight='bold')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='red', alpha=0.7, s=60, label='Franz Schubert (Different)'),
        plt.scatter([], [], c='blue', alpha=0.7, s=60, label='Same Domain'),
        plt.scatter([], [], c='gray', alpha=0.7, s=60, label='Different Domain')
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    # 3. Clustering visualization
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Simulate 2D projection of embeddings using t-SNE-like positioning
    positions = {
        'Schubert, Franz (Composer)': (0.3, 0.7),
        'Schubert, Franz (Photographer)': (0.8, 0.3),
        'Bach, Johann Sebastian': (0.1, 0.8),
        'García Márquez, Gabriel': (0.7, 0.8),
        'Einstein, Albert': (0.9, 0.9),
        'Adams, Ansel (Photographer)': (0.9, 0.2),
        'Mozart, Wolfgang Amadeus': (0.2, 0.9),
        'Neruda, Pablo': (0.8, 0.7)
    }
    
    # Define clusters
    clusters = {
        'Music': ['Schubert, Franz (Composer)', 'Bach, Johann Sebastian', 'Mozart, Wolfgang Amadeus'],
        'Literature': ['García Márquez, Gabriel', 'Neruda, Pablo'],
        'Photography': ['Schubert, Franz (Photographer)', 'Adams, Ansel (Photographer)'],
        'Science': ['Einstein, Albert']
    }
    
    cluster_colors = {'Music': '#3498DB', 'Literature': '#27AE60', 
                     'Photography': '#9B59B6', 'Science': '#E74C3C'}
    
    # Plot points and cluster boundaries
    for cluster_name, entities_in_cluster in clusters.items():
        if len(entities_in_cluster) > 1:
            # Get positions for this cluster
            cluster_positions = [positions[entity] for entity in entities_in_cluster]
            xs, ys = zip(*cluster_positions)
            
            # Draw cluster boundary (convex hull approximation)
            from matplotlib.patches import Ellipse
            center_x, center_y = np.mean(xs), np.mean(ys)
            width, height = (max(xs) - min(xs) + 0.2), (max(ys) - min(ys) + 0.2)
            ellipse = Ellipse((center_x, center_y), width, height,
                            facecolor=cluster_colors[cluster_name], alpha=0.2,
                            edgecolor=cluster_colors[cluster_name], linewidth=2)
            ax3.add_patch(ellipse)
            
            # Add cluster label
            ax3.text(center_x, center_y - height/2 - 0.05, cluster_name,
                    ha='center', va='top', fontweight='bold', 
                    color=cluster_colors[cluster_name])
    
    # Plot entity points
    for entity, (x, y) in positions.items():
        # Determine cluster color
        entity_color = 'black'
        for cluster_name, entities_in_cluster in clusters.items():
            if entity in entities_in_cluster:
                entity_color = cluster_colors[cluster_name]
                break
        
        ax3.scatter(x, y, s=100, c=entity_color, alpha=0.8, 
                   edgecolor='white', linewidth=2)
        
        # Add label (shortened)
        short_name = entity.split(',')[0]
        if 'Photographer' in entity:
            short_name += ' (P)'
        elif 'Composer' in entity:
            short_name += ' (C)'
            
        ax3.text(x, y-0.08, short_name, ha='center', va='top', 
                fontsize=8, fontweight='bold')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Entity Clustering in Vector Space\n(2D projection of high-dimensional embeddings)', 
                 fontweight='bold')
    ax3.axis('off')
    
    # Add insight note
    insight_text = (
        "🔍 KEY OBSERVATIONS:\n"
        "• Franz Schuberts cluster separately by domain\n"
        "• High similarity within domains (Bach-Mozart: 0.89)\n"
        "• Cross-domain similarities are lower\n"
        "• Vector space naturally groups related entities"
    )
    
    ax3.text(0.02, 0.02, insight_text, transform=ax3.transAxes,
            ha='left', va='bottom', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFF3CD', 
                     edgecolor='#F39C12', linewidth=2))
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the similarity heatmaps visualization"""
    
    # Create the visualization
    fig = create_similarity_heatmaps()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "similarity_heatmap.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Similarity heatmaps saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()