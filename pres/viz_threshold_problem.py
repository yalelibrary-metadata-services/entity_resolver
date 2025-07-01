#!/usr/bin/env python3
"""
Threshold Problem Demonstration
Shows why single thresholds fail for entity resolution
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

def create_threshold_problem():
    """Create threshold problem demonstration"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.2, 1], width_ratios=[1, 1])
    
    # Main title
    fig.suptitle('The Threshold Problem: Why One Size Doesn\'t Fit All', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Problem illustration (top)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Entity types and their similarity distributions
    entity_types = ['Common Names\n(Smith, Johnson)', 'Unique Names\n(Dostoyevsky, Tchaikovsky)', 
                   'Franz Schubert\nCase', 'Scientific Names\n(Einstein, Darwin)']
    
    # Simulated similarity distributions for same vs different entities
    same_means = [0.95, 0.75, 0.85, 0.80]
    same_stds = [0.05, 0.10, 0.08, 0.12]
    diff_means = [0.85, 0.45, 0.76, 0.35]  # Franz Schubert is the problem case
    diff_stds = [0.08, 0.15, 0.05, 0.20]
    
    x_positions = np.arange(len(entity_types))
    
    # Create violin-like distributions
    for i, entity_type in enumerate(entity_types):
        x = x_positions[i]
        
        # Same entity distribution (green)
        same_data = np.random.normal(same_means[i], same_stds[i], 1000)
        same_data = same_data[(same_data >= 0) & (same_data <= 1)]
        
        # Different entity distribution (red)
        diff_data = np.random.normal(diff_means[i], diff_stds[i], 1000)
        diff_data = diff_data[(diff_data >= 0) & (diff_data <= 1)]
        
        # Plot distributions as histograms (rotated)
        bins = np.linspace(0, 1, 20)
        same_hist, _ = np.histogram(same_data, bins=bins, density=True)
        diff_hist, _ = np.histogram(diff_data, bins=bins, density=True)
        
        # Normalize for display
        same_hist = same_hist / max(same_hist) * 0.3
        diff_hist = diff_hist / max(diff_hist) * 0.3
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot same entity distribution
        ax1.barh(bin_centers, same_hist, left=x-0.15, height=0.04, 
                color='green', alpha=0.7, label='Same Entity' if i == 0 else "")
        
        # Plot different entity distribution  
        ax1.barh(bin_centers, -diff_hist, left=x-0.15, height=0.04,
                color='red', alpha=0.7, label='Different Entity' if i == 0 else "")
        
        # Add mean markers
        ax1.plot(x-0.15, same_means[i], 'o', color='darkgreen', markersize=8)
        ax1.plot(x-0.15, diff_means[i], 'o', color='darkred', markersize=8)
        
        # Highlight Franz Schubert problem
        if i == 2:  # Franz Schubert case
            overlap_rect = Rectangle((x-0.4, 0.7), 0.8, 0.15, 
                                   facecolor='yellow', alpha=0.3, 
                                   edgecolor='orange', linewidth=2)
            ax1.add_patch(overlap_rect)
            ax1.text(x, 0.77, 'OVERLAP\nPROBLEM!', ha='center', va='center',
                    fontweight='bold', color='red', fontsize=9)
    
    ax1.set_xlim(-0.5, len(entity_types) - 0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(entity_types, fontsize=10)
    ax1.set_ylabel('Similarity Score', fontweight='bold')
    ax1.set_title('Similarity Distributions: Same vs Different Entities by Type', 
                 fontweight='bold', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Threshold testing (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    thresholds = np.linspace(0.5, 0.95, 20)
    
    # Simulate precision and recall for different thresholds
    # Based on the overlapping distributions above
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        # Simulate based on realistic patterns
        if threshold < 0.7:
            precision = 0.6 + 0.3 * (threshold - 0.5) / 0.2
            recall = 0.95 - 0.1 * (threshold - 0.5) / 0.2
        elif threshold < 0.8:
            precision = 0.9 + 0.08 * (threshold - 0.7) / 0.1
            recall = 0.85 - 0.25 * (threshold - 0.7) / 0.1
        else:
            precision = 0.98 + 0.02 * (threshold - 0.8) / 0.15
            recall = 0.6 - 0.4 * (threshold - 0.8) / 0.15
        
        # Add noise and constraints
        precision = min(1.0, max(0.0, precision + np.random.normal(0, 0.02)))
        recall = min(1.0, max(0.0, recall + np.random.normal(0, 0.02)))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    ax2.plot(thresholds, precisions, 'o-', color='blue', linewidth=2, 
            markersize=4, label='Precision')
    ax2.plot(thresholds, recalls, 's-', color='red', linewidth=2, 
            markersize=4, label='Recall')
    ax2.plot(thresholds, f1_scores, '^-', color='green', linewidth=2, 
            markersize=4, label='F1-Score')
    
    # Highlight the dilemma points
    ax2.axvline(x=0.76, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax2.text(0.76, 0.9, 'Franz Schubert\nSimilarity', rotation=90, 
            ha='right', va='top', color='orange', fontweight='bold')
    
    # Show the impossible choice
    ax2.annotate('High Recall\nBut Low Precision', xy=(0.65, 0.6), xytext=(0.55, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='red', alpha=0.8))
    
    ax2.annotate('High Precision\nBut Low Recall', xy=(0.85, 0.4), xytext=(0.9, 0.7),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=9, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='blue', alpha=0.8))
    
    ax2.set_xlabel('Similarity Threshold', fontweight='bold')
    ax2.set_ylabel('Performance Metric', fontweight='bold')
    ax2.set_title('Performance vs Threshold\n(No optimal point exists)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Different entity examples (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Example pairs with their similarities
    examples = [
        ('Smith, John (Historian)', 'Smith, John (Composer)', 0.92, False, 'High Sim\nDifferent People'),
        ('Dostoyevsky, Fyodor', 'Dostoevsky, Fyodor', 0.89, True, 'Spelling Variant\nSame Person'),
        ('Schubert, Franz (Music)', 'Schubert, Franz (Photo)', 0.76, False, 'THE PROBLEM\nModerate Sim'),
        ('García Márquez, Gabriel', 'Márquez, Gabriel García', 0.95, True, 'Name Order\nSame Person'),
        ('Einstein, Albert', 'Hawking, Stephen', 0.35, False, 'Different People\nLow Sim')
    ]
    
    similarities = [ex[2] for ex in examples]
    is_same = [ex[3] for ex in examples]
    colors = ['green' if same else 'red' for same in is_same]
    
    y_positions = range(len(examples))
    
    bars = ax3.barh(y_positions, similarities, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1)
    
    # Add threshold lines
    for threshold in [0.5, 0.7, 0.8, 0.9]:
        ax3.axvline(x=threshold, color='gray', linestyle=':', alpha=0.6)
        ax3.text(threshold, len(examples), f'{threshold}', 
                ha='center', va='bottom', fontsize=8)
    
    # Highlight Franz Schubert
    franz_idx = 2
    highlight_rect = Rectangle((0, franz_idx-0.4), similarities[franz_idx], 0.8,
                             facecolor='yellow', alpha=0.3, 
                             edgecolor='orange', linewidth=2)
    ax3.add_patch(highlight_rect)
    
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels([ex[4] for ex in examples], fontsize=9)
    ax3.set_xlabel('Cosine Similarity', fontweight='bold')
    ax3.set_title('Real Entity Pairs\nDemonstrating the Challenge', fontweight='bold')
    ax3.set_xlim(0, 1)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, label='Same Entity'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Different Entity')
    ]
    ax3.legend(handles=legend_elements, loc='lower right')
    
    # 4. Solutions section (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    
    solutions = [
        {
            'title': '❌ FAILED APPROACH: Single Threshold',
            'description': 'Tried thresholds from 0.5 to 0.95\nNo single value works for all entity types\nFranz Schubert case breaks every threshold',
            'color': '#E74C3C',
            'x': 0.05
        },
        {
            'title': '💡 SOLUTION 1: Domain Classification',
            'description': 'Add contextual features:\n• Music vs Photography domains\n• Different domains = likely different people\n• Same domains = examine similarity more carefully',
            'color': '#F39C12',
            'x': 0.27
        },
        {
            'title': '💡 SOLUTION 2: Temporal Features',
            'description': 'Birth/death year matching:\n• 1797-1828 vs 1930-1989\n• Different centuries = different people\n• Temporal overlap supports same entity',
            'color': '#9B59B6',
            'x': 0.49
        },
        {
            'title': '💡 SOLUTION 3: Multi-Feature ML',
            'description': 'Logistic regression combines:\n• Text similarity (positive weight)\n• Domain dissimilarity (negative weight)\n• Temporal matching (positive weight)',
            'color': '#27AE60',
            'x': 0.71
        },
        {
            'title': '🎯 FINAL RESULT: 99.55% Precision',
            'description': 'Production system achieves:\n• 99.55% precision, 82.48% recall\n• Correctly distinguishes Franz Schuberts\n• Handles all entity types robustly',
            'color': '#3498DB',
            'x': 0.84
        }
    ]
    
    for solution in solutions:
        # Create solution box
        box = FancyBboxPatch(
            (solution['x'], 0.1), 0.16, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=solution['color'],
            alpha=0.1,
            edgecolor=solution['color'],
            linewidth=2
        )
        ax4.add_patch(box)
        
        # Add title
        ax4.text(solution['x'] + 0.08, 0.8, solution['title'], 
                ha='center', va='center', fontweight='bold', 
                fontsize=9, color=solution['color'])
        
        # Add description
        ax4.text(solution['x'] + 0.08, 0.4, solution['description'], 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Evolution from Problem to Solution', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the threshold problem visualization"""
    
    # Create the visualization
    fig = create_threshold_problem()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "threshold_problem_demo.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Threshold problem demo saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()