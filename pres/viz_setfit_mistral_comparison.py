#!/usr/bin/env python3
"""
SetFit vs Mistral Comparison Visualization
Shows capability comparison and decision matrix
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

def create_comparison_matrix():
    """Create SetFit vs Mistral comparison matrix"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.5, 0.7], width_ratios=[1.2, 0.8])
    
    # Main title
    fig.suptitle('SetFit vs Mistral Classifier Factory: The Pivot Decision', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Comparison matrix (main visualization)
    ax1 = fig.add_subplot(gs[1, :])
    
    # Comparison data
    criteria = [
        'Token Limit',
        'Context Length', 
        'Training Required',
        'Inference Speed',
        'Few-shot Learning',
        'Multilingual Support',
        'Model Size',
        'Cost per Classification',
        'Setup Complexity',
        'Customization',
        'Yale Compatibility',
        'Production Ready'
    ]
    
    setfit_scores = [
        '128 tokens',
        '~500 chars',
        'Yes (5-10 min)',
        'Very Fast',
        'Excellent',
        'Good',
        '110M params',
        '$0 (after training)',
        'Moderate',
        'High',
        '❌ 25% data fits',
        '⚠️ Limited by tokens'
    ]
    
    mistral_scores = [
        '32K tokens',
        '~128K chars',
        'No training',
        'Fast',
        'Excellent',
        'Excellent',
        '7B params',
        '~$0.001 per call',
        'Simple',
        'Moderate', 
        '✅ 100% data fits',
        '✅ Production scale'
    ]
    
    # Color coding for comparison
    setfit_colors = ['#E74C3C', '#E74C3C', '#F39C12', '#27AE60', '#27AE60', 
                     '#F39C12', '#3498DB', '#27AE60', '#F39C12', '#27AE60',
                     '#E74C3C', '#F39C12']
    
    mistral_colors = ['#27AE60', '#27AE60', '#27AE60', '#27AE60', '#27AE60',
                      '#27AE60', '#E74C3C', '#F39C12', '#27AE60', '#F39C12',
                      '#27AE60', '#27AE60']
    
    # Create comparison table
    table_data = []
    for i, criterion in enumerate(criteria):
        table_data.append([criterion, setfit_scores[i], mistral_scores[i]])
    
    # Create visual table
    cell_height = 0.6
    cell_width = 2.5
    start_x = 2
    start_y = len(criteria) * cell_height
    
    # Headers
    headers = ['Criterion', 'SetFit', 'Mistral Classifier Factory']
    header_colors = ['#34495E', '#9B59B6', '#E67E22']
    
    for j, (header, color) in enumerate(zip(headers, header_colors)):
        rect = Rectangle((start_x + j * cell_width, start_y), cell_width, cell_height,
                        facecolor=color, edgecolor='white', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(start_x + j * cell_width + cell_width/2, start_y + cell_height/2,
                header, ha='center', va='center', fontweight='bold', 
                color='white', fontsize=11)
    
    # Data rows
    for i, (criterion, setfit_val, mistral_val) in enumerate(table_data):
        y = start_y - (i + 1) * cell_height
        
        # Criterion column
        rect = Rectangle((start_x, y), cell_width, cell_height,
                        facecolor='#ECF0F1', edgecolor='white', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(start_x + cell_width/2, y + cell_height/2,
                criterion, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # SetFit column
        rect = Rectangle((start_x + cell_width, y), cell_width, cell_height,
                        facecolor=setfit_colors[i], alpha=0.3,
                        edgecolor='white', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(start_x + cell_width + cell_width/2, y + cell_height/2,
                setfit_val, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Mistral column
        rect = Rectangle((start_x + 2*cell_width, y), cell_width, cell_height,
                        facecolor=mistral_colors[i], alpha=0.3,
                        edgecolor='white', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(start_x + 2*cell_width + cell_width/2, y + cell_height/2,
                mistral_val, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    legend_y = start_y - len(criteria) * cell_height - 1
    legend_elements = [
        ('✅ Advantage', '#27AE60'),
        ('⚠️ Acceptable', '#F39C12'), 
        ('❌ Disadvantage', '#E74C3C')
    ]
    
    for i, (label, color) in enumerate(legend_elements):
        rect = Rectangle((start_x + i * 2.5, legend_y), 0.3, 0.3,
                        facecolor=color, alpha=0.3, edgecolor=color)
        ax1.add_patch(rect)
        ax1.text(start_x + i * 2.5 + 0.5, legend_y + 0.15,
                label, ha='left', va='center', fontsize=9)
    
    ax1.set_xlim(0, 12)
    ax1.set_ylim(legend_y - 0.5, start_y + 2)
    ax1.axis('off')
    
    # 2. Decision timeline (top)
    ax2 = fig.add_subplot(gs[0, :])
    
    timeline_events = [
        ('Initial Research', 1, '#3498DB', 'SetFit looks perfect\nfor few-shot learning'),
        ('Reality Check', 3, '#F39C12', 'Token limits discovered\nwith real data'),
        ('Architecture Pivot', 5, '#E74C3C', 'SetFit can only handle\n25% of Yale data'),
        ('Mistral Solution', 7, '#27AE60', 'Handles 100% of data\nwith full context')
    ]
    
    # Draw timeline
    ax2.plot([0.5, 7.5], [0.5, 0.5], 'k-', linewidth=3, alpha=0.3)
    
    for event, x, color, description in timeline_events:
        # Event point
        ax2.plot(x, 0.5, 'o', markersize=15, color=color, 
                markeredgecolor='white', markeredgewidth=2)
        
        # Event label
        ax2.text(x, 0.8, event, ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color=color)
        
        # Description
        ax2.text(x, 0.2, description, ha='center', va='top', 
                fontsize=8, style='italic',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor=color, alpha=0.8))
    
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 1.2)
    ax2.axis('off')
    ax2.set_title('Decision Timeline: From SetFit to Mistral', 
                 fontweight='bold', pad=20)
    
    # 3. Key insights (bottom)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off')
    
    insights_text = [
        "🔍 LESSON LEARNED: Always test with realistic data, not toy examples",
        "⚖️ TRADE-OFFS: SetFit's speed vs Mistral's context handling capability", 
        "🎯 DECISION DRIVER: Token length compatibility was non-negotiable",
        "💡 OUTCOME: Mistral enabled 100% data coverage with superior context"
    ]
    
    # Create insight boxes
    box_width = 0.22
    for i, insight in enumerate(insights_text):
        x = 0.02 + i * 0.24
        
        # Determine color based on content
        if 'LESSON' in insight:
            color = '#E74C3C'
        elif 'TRADE-OFFS' in insight:
            color = '#F39C12'
        elif 'DECISION' in insight:
            color = '#9B59B6'
        else:
            color = '#27AE60'
        
        rect = Rectangle((x, 0.2), box_width, 0.6, 
                        facecolor=color, alpha=0.1,
                        edgecolor=color, linewidth=2)
        ax3.add_patch(rect)
        
        ax3.text(x + box_width/2, 0.5, insight, 
                ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Add final verdict box
    verdict_text = (
        "🏆 WINNER: MISTRAL CLASSIFIER FACTORY\n\n"
        "Key factors:\n"
        "• 32K token limit vs 128 tokens\n"
        "• 100% data compatibility\n" 
        "• No training overhead\n"
        "• Production-ready scale\n\n"
        "Cost: $18K vs $1.76M manual\nROI: 99% cost reduction"
    )
    
    ax1.text(10, 4, verdict_text,
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=3))
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the SetFit vs Mistral comparison"""
    
    # Create the visualization
    fig = create_comparison_matrix()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "setfit_vs_mistral_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ SetFit vs Mistral comparison saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()