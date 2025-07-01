#!/usr/bin/env python3
"""
Journey Timeline Visualization
Shows the evolution from simple embeddings to production system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# Set up professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def create_journey_timeline():
    """Create the complete journey timeline visualization"""
    
    # Create figure with generous size for readability
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Timeline data: (phase, x_position, y_position, problem, solution, color)
    timeline_phases = [
        {
            'phase': '1. EMBEDDINGS\nFUNDAMENTALS',
            'x': 1,
            'y': 4,
            'problem': 'Franz Schubert Problem',
            'solution': 'Text embeddings\nfor similarity',
            'discovery': 'Threshold problem:\nNo universal cutoff works',
            'color': '#3498DB',  # Blue
            'notebook': 'Notebook 1'
        },
        {
            'phase': '2. DOMAIN\nCLASSIFICATION',
            'x': 3,
            'y': 4,
            'problem': 'Context missing\nfor disambiguation',
            'solution': 'Domain taxonomy\nwith classification',
            'discovery': 'SetFit token limits\nforced Mistral pivot',
            'color': '#E74C3C',  # Red
            'notebook': 'Notebook 2'
        },
        {
            'phase': '3. PRODUCTION\nPIPELINE',
            'x': 5,
            'y': 4,
            'problem': 'Scale + missing data\n+ feature integration',
            'solution': 'Weaviate + hot-deck\n+ ML classification',
            'discovery': '99.55% precision\n17.6M records',
            'color': '#27AE60',  # Green
            'notebook': 'Notebook 3'
        }
    ]
    
    # Draw timeline backbone
    ax.plot([0.5, 5.5], [4, 4], 'k-', linewidth=3, alpha=0.3)
    
    # Add phase boxes and details
    for phase in timeline_phases:
        x, y = phase['x'], phase['y']
        color = phase['color']
        
        # Main phase box
        phase_box = FancyBboxPatch(
            (x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(phase_box)
        
        # Phase label
        ax.text(x, y, phase['phase'], 
                ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        # Notebook indicator
        ax.text(x, y+0.5, phase['notebook'], 
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)
        
        # Problem box (above)
        problem_box = FancyBboxPatch(
            (x-0.5, y+1.0), 1.0, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#F8F9FA',
            edgecolor=color,
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(problem_box)
        
        ax.text(x, y+1.4, '🚧 PROBLEM', 
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#E74C3C')
        ax.text(x, y+1.1, phase['problem'], 
                ha='center', va='center',
                fontsize=9, color='#2C3E50')
        
        # Solution box (below)
        solution_box = FancyBboxPatch(
            (x-0.5, y-1.8), 1.0, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#F8F9FA',
            edgecolor=color,
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(solution_box)
        
        ax.text(x, y-1.6, '💡 SOLUTION', 
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#27AE60')
        ax.text(x, y-1.9, phase['solution'], 
                ha='center', va='center',
                fontsize=9, color='#2C3E50')
        
        # Discovery box (far below)
        discovery_box = FancyBboxPatch(
            (x-0.5, y-3.0), 1.0, 0.8,
            boxstyle="round,pad=0.05",
            facecolor='#FFF3CD',
            edgecolor='#F39C12',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(discovery_box)
        
        ax.text(x, y-2.8, '🔍 DISCOVERY', 
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#F39C12')
        ax.text(x, y-3.1, phase['discovery'], 
                ha='center', va='center',
                fontsize=9, color='#D35400')
        
        # Timeline point
        ax.plot(x, y, 'o', markersize=12, color='white', 
                markeredgecolor=color, markeredgewidth=3)
    
    # Add arrows between phases
    for i in range(len(timeline_phases) - 1):
        x1 = timeline_phases[i]['x'] + 0.4
        x2 = timeline_phases[i+1]['x'] - 0.4
        y = 4
        
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))
    
    # Add Franz Schubert thread
    franz_y = 5.5
    ax.text(3, franz_y, '🎼 Franz Schubert Thread', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F4F8', 
                     edgecolor='#3498DB', linewidth=2))
    
    # Add Franz Schubert examples at each phase
    franz_examples = [
        "Different Franz Schuberts\nscore 0.76 similarity",
        "Composer → Music domain\nPhotographer → Documentation",
        "System correctly distinguishes\nboth entities"
    ]
    
    for i, (phase, example) in enumerate(zip(timeline_phases, franz_examples)):
        x = phase['x']
        ax.plot([x, x], [franz_y-0.3, y+0.3], '--', color='#3498DB', alpha=0.5)
        ax.text(x, franz_y-0.8, example, 
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                         edgecolor='#3498DB', alpha=0.8))
    
    # Add title and description
    fig.suptitle('From Words to Vectors: The Entity Resolution Journey', 
                fontsize=18, fontweight='bold', y=0.95)
    
    ax.text(3, 0.5, 
            'A chronological journey through building Yale University Library\'s\n'
            'entity resolution system: from simple embeddings to production AI\n'
            'processing 17.6 million catalog records with 99.55% precision',
            ha='center', va='center', fontsize=12, style='italic',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#F8F9FA', 
                     edgecolor='#BDC3C7', linewidth=1))
    
    # Add key insights box
    insights_text = (
        "🎯 KEY INSIGHTS:\n"
        "• Real AI development is iterative and problem-driven\n"
        "• Each solution reveals new challenges requiring innovation\n"
        "• Production systems combine multiple techniques\n"
        "• Domain expertise drives feature engineering decisions"
    )
    
    ax.text(6.5, 2, insights_text,
            ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=2))
    
    # Set axis limits and remove axes
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-4, 6.5)
    ax.axis('off')
    
    # Add subtle grid for reference
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the journey timeline visualization"""
    
    # Create the visualization
    fig = create_journey_timeline()
    
    # Save to img directory
    output_path = Path(__file__).parent / "img" / "journey_timeline.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Journey timeline saved to {output_path}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()