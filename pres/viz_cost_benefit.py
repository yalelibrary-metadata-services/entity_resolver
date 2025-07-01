#!/usr/bin/env python3
"""
Cost-Benefit Analysis Visualization
Shows economic impact of automated vs manual entity resolution
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_cost_benefit():
    """Create cost-benefit analysis visualization"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8], width_ratios=[1, 1, 1])
    
    fig.suptitle('Economic Analysis: ROI of Automated Entity Resolution', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Cost comparison (main chart)
    ax1 = fig.add_subplot(gs[0, :2])
    
    categories = ['Embedding\nCosts', 'Classification\nCosts', 'Infrastructure\nCosts', 'Manual Review\nCosts', 'TOTAL']
    
    # Automated system costs
    automated_costs = [26400, 18000, 5000, 7500, 56900]  # Batch + Mistral + Weaviate + reduced manual
    
    # Manual system costs  
    manual_costs = [0, 0, 2000, 1760000, 1762000]  # Minimal tech + full manual review
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, automated_costs, width, label='Automated System', 
                   color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, manual_costs, width, label='Manual Process', 
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars, costs in [(bars1, automated_costs), (bars2, manual_costs)]:
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10000,
                    f'${cost:,.0f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Cost (USD)', fontweight='bold')
    ax1.set_xlabel('Cost Category', fontweight='bold')
    ax1.set_title('Cost Breakdown: Automated vs Manual Entity Resolution\n(17.6M Yale catalog records)', 
                 fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 2000000)
    
    # Highlight savings
    savings = manual_costs[-1] - automated_costs[-1]
    roi_percentage = (savings / automated_costs[-1]) * 100
    
    ax1.text(0.7, 0.8, f'💰 SAVINGS: ${savings:,.0f}\n📈 ROI: {roi_percentage:.0f}%', 
            transform=ax1.transAxes, ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#E8F8F5', 
                     edgecolor='#27AE60', linewidth=3))
    
    # 2. Time comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Time estimates
    tasks = ['Setup', 'Processing', 'Review', 'Total']
    automated_time = [2, 3, 1, 6]  # weeks
    manual_time = [0.5, 0, 104, 104.5]  # weeks (2 years)
    
    y_pos = np.arange(len(tasks))
    
    bars_auto = ax2.barh(y_pos - 0.2, automated_time, 0.4, label='Automated', 
                        color='#3498DB', alpha=0.8)
    bars_manual = ax2.barh(y_pos + 0.2, manual_time, 0.4, label='Manual', 
                          color='#E67E22', alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tasks)
    ax2.set_xlabel('Time (Weeks)', fontweight='bold')
    ax2.set_title('Time Comparison\nProject Duration', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add time labels
    for bars, times in [(bars_auto, automated_time), (bars_manual, manual_time)]:
        for bar, time in zip(bars, times):
            width = bar.get_width()
            if width > 0:
                ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                        f'{time}w', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 3. ROI over time (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    years = np.arange(0, 6)
    
    # Cumulative costs over time
    automated_cumulative = [56900 + year * 15000 for year in years]  # Initial + maintenance
    manual_cumulative = [1762000 + year * 350000 for year in years]  # Initial + ongoing
    
    ax3.plot(years, automated_cumulative, 'o-', color='#27AE60', linewidth=3, 
            markersize=8, label='Automated System')
    ax3.plot(years, manual_cumulative, 's-', color='#E74C3C', linewidth=3, 
            markersize=8, label='Manual Process')
    
    ax3.set_xlabel('Years', fontweight='bold')
    ax3.set_ylabel('Cumulative Cost (USD)', fontweight='bold')
    ax3.set_title('5-Year Cost Projection', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='plain', axis='y')
    
    # Highlight breakeven
    ax3.text(2, 2000000, 'Immediate\nBreakeven', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#27AE60',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                     edgecolor='#27AE60', linewidth=2))
    
    # 4. Quality benefits (bottom center)
    ax4 = fig.add_subplot(gs[1, 1])
    
    metrics = ['Precision', 'Recall', 'Consistency', 'Scalability', 'Speed']
    automated_quality = [99.55, 82.48, 100, 100, 100]
    manual_quality = [85, 70, 60, 20, 10]  # Estimates for manual process
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    automated_quality += automated_quality[:1]
    manual_quality += manual_quality[:1]
    
    ax4 = plt.subplot(gs[1, 1], projection='polar')
    ax4.plot(angles, automated_quality, 'o-', linewidth=2, color='#27AE60', label='Automated')
    ax4.fill(angles, automated_quality, alpha=0.25, color='#27AE60')
    ax4.plot(angles, manual_quality, 's-', linewidth=2, color='#E74C3C', label='Manual')
    ax4.fill(angles, manual_quality, alpha=0.25, color='#E74C3C')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 100)
    ax4.set_title('Quality Comparison\n(% Performance)', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 5. Business impact summary (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    impact_summary = (
        "📊 BUSINESS IMPACT SUMMARY\n\n"
        "💰 FINANCIAL BENEFITS:\n"
        f"• Cost reduction: 97%\n"
        f"• Annual savings: ${savings:,.0f}\n"
        f"• ROI: {roi_percentage:.0f}% (immediate)\n"
        f"• Payback period: < 1 month\n\n"
        "⚡ OPERATIONAL BENEFITS:\n"
        "• 17x faster processing\n"
        "• 99.55% precision vs ~85% manual\n"
        "• 24/7 availability\n"
        "• Consistent quality\n"
        "• Scalable to any size\n\n"
        "🚀 STRATEGIC BENEFITS:\n"
        "• Enables new services\n"
        "• Frees experts for complex work\n"
        "• Real-time catalog enhancement\n"
        "• Foundation for AI initiatives\n"
        "• Competitive advantage"
    )
    
    ax5.text(0.05, 0.95, impact_summary, 
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#F8F9FA', 
                     edgecolor='#2C3E50', linewidth=2),
            transform=ax5.transAxes)
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the cost-benefit analysis"""
    
    fig = create_cost_benefit()
    
    output_path = Path(__file__).parent / "img" / "cost_benefit_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Cost-benefit analysis saved to {output_path}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()