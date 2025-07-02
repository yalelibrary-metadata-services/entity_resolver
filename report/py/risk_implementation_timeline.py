"""
Risk Assessment and Implementation Timeline Visualization

Strategic Purpose: Martin Kurth needs to see that we've thought through what could go wrong
and have concrete plans to mitigate risks. The implementation timeline shows how we reduce
exposure through phased deployment rather than "big bang" implementation. This addresses
the conservative administrator's primary concern: "What if this doesn't work?"

Key Teaching Points:
- Risk matrices help visualize probability vs impact tradeoffs
- Phased implementation reduces cumulative risk exposure  
- Mitigation strategies show proactive planning rather than wishful thinking
- Timeline visualization helps with resource planning and milestone tracking

Key Message: We've planned for what could go wrong and structured implementation 
to minimize risk while maximizing learning opportunities.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_risk_assessment_matrix():
    """
    Creates a comprehensive risk assessment matrix that shows how we've
    categorized and planned for different types of risks.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Risk Probability vs Impact Matrix
    # Define risks with their probability and impact assessments
    risks = {
        'Vendor Dependencies': {'prob': 25, 'impact': 40, 'category': 'Technical'},
        'Staff Acceptance': {'prob': 30, 'impact': 35, 'category': 'Operational'},
        'Data Quality Issues': {'prob': 40, 'impact': 30, 'category': 'Technical'},
        'Cost Escalation': {'prob': 20, 'impact': 45, 'category': 'Financial'},
        'Performance Degradation': {'prob': 15, 'impact': 50, 'category': 'Technical'},
        'Integration Complexity': {'prob': 35, 'impact': 25, 'category': 'Technical'},
        'Training Inadequacy': {'prob': 25, 'impact': 20, 'category': 'Operational'},
        'Quality Control Lapses': {'prob': 20, 'impact': 60, 'category': 'Operational'},
        'Timeline Delays': {'prob': 45, 'impact': 30, 'category': 'Project'},
        'Stakeholder Resistance': {'prob': 15, 'impact': 35, 'category': 'Political'}
    }
    
    # Create scatter plot with different colors for categories
    category_colors = {'Technical': '#3b82f6', 'Operational': '#10b981', 
                      'Financial': '#ef4444', 'Project': '#f59e0b', 'Political': '#8b5cf6'}
    
    for risk_name, data in risks.items():
        ax1.scatter(data['prob'], data['impact'], 
                   c=category_colors[data['category']], 
                   s=150, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add risk labels with slight offset to avoid overlap
        ax1.annotate(risk_name, (data['prob'], data['impact']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add risk level zones
    # Low risk (green zone)
    low_risk = Rectangle((0, 0), 30, 30, alpha=0.2, facecolor='green', label='Low Risk')
    ax1.add_patch(low_risk)
    
    # Medium risk (yellow zone)  
    med_risk = Rectangle((30, 0), 40, 50, alpha=0.2, facecolor='yellow', label='Medium Risk')
    ax1.add_patch(med_risk)
    med_risk2 = Rectangle((0, 30), 30, 40, alpha=0.2, facecolor='yellow')
    ax1.add_patch(med_risk2)
    
    # High risk (red zone)
    high_risk = Rectangle((30, 50), 70, 50, alpha=0.2, facecolor='red', label='High Risk')
    ax1.add_patch(high_risk)
    high_risk2 = Rectangle((70, 0), 30, 50, alpha=0.2, facecolor='red')
    ax1.add_patch(high_risk2)
    
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Impact Level', fontsize=12, fontweight='bold')
    ax1.set_title('Risk Assessment Matrix', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Create legend for categories
    legend_elements = [plt.scatter([], [], c=color, s=100, label=cat) 
                      for cat, color in category_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper left', title='Risk Categories')
    
    # 2. Risk Mitigation Strategies
    ax2.axis('off')
    
    mitigation_text = """
RISK MITIGATION STRATEGIES

🔧 Technical Risks:
• Vendor Dependencies: Standard APIs enable provider migration
• Data Quality: Confidence scoring flags questionable decisions  
• Performance: Conservative scaling with monitoring thresholds
• Integration: Modular architecture with rollback capabilities

👥 Operational Risks:
• Staff Acceptance: 40-hour training program + gradual deployment
• Quality Control: Multi-layer QA with human review checkpoints
• Training: Hands-on workshops with expert cataloger involvement

💰 Financial Risks:
• Cost Escalation: 30% contingency budget + phased funding
• Contracts: Fixed-price APIs with alternative provider options

⏱️ Project Risks:
• Timeline Delays: Conservative estimates + parallel workstreams
• Dependencies: Early identification with mitigation plans

🏛️ Political Risks:
• Stakeholder Resistance: Transparency + demonstrated benefits
• Change Management: Involving staff in system design decisions
"""
    
    ax2.text(0.05, 0.95, mitigation_text, transform=ax2.transAxes, fontsize=10,
            va='top', ha='left', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#f0fdf4', alpha=0.8))
    
    # 3. Risk Timeline (how risks change during implementation)
    phases = ['Planning', 'Phase 1', 'Phase 2', 'Phase 3', 'Operations']
    
    # Risk levels by phase for key risks
    vendor_risk = [30, 25, 20, 15, 10]
    staff_risk = [40, 35, 25, 15, 10]
    quality_risk = [50, 40, 30, 20, 15]
    
    x_pos = np.arange(len(phases))
    width = 0.25
    
    ax3.bar(x_pos - width, vendor_risk, width, label='Vendor Dependencies', 
           color='#3b82f6', alpha=0.8)
    ax3.bar(x_pos, staff_risk, width, label='Staff Acceptance', 
           color='#10b981', alpha=0.8)
    ax3.bar(x_pos + width, quality_risk, width, label='Quality Control', 
           color='#ef4444', alpha=0.8)
    
    ax3.set_xlabel('Implementation Phase', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Risk Level', fontsize=12, fontweight='bold')
    ax3.set_title('Risk Evolution During Implementation', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(phases)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cumulative Risk Exposure
    # Calculate cumulative risk score over time
    total_risk_by_phase = [sum(x) for x in zip(vendor_risk, staff_risk, quality_risk)]
    cumulative_exposure = np.cumsum(total_risk_by_phase)
    
    # Compare to "big bang" implementation approach
    big_bang_risk = [150] * 5  # Constant high risk
    big_bang_cumulative = np.cumsum(big_bang_risk)
    
    ax4.plot(phases, cumulative_exposure, 'g-o', linewidth=3, markersize=8,
            label='Phased Implementation')
    ax4.plot(phases, big_bang_cumulative, 'r--s', linewidth=3, markersize=8,
            label='Big Bang Approach')
    
    ax4.set_xlabel('Implementation Phase', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Risk Exposure', fontsize=12, fontweight='bold')
    ax4.set_title('Risk Exposure: Phased vs Big Bang Implementation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Highlight the difference
    savings_at_end = big_bang_cumulative[-1] - cumulative_exposure[-1]
    ax4.annotate(f'Risk Reduction:\n{savings_at_end:.0f} points', 
                xy=(4, cumulative_exposure[-1]), xytext=(3, 400),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                fontsize=11, fontweight='bold')
    
    plt.suptitle('Risk Assessment and Mitigation Strategy', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_implementation_timeline():
    """
    Creates a detailed implementation timeline that shows how the phased
    approach manages risk while building capabilities incrementally.
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Define implementation timeline
    start_date = datetime(2025, 8, 1)  # August 2025 start
    
    # Phase definitions with start dates and durations (in weeks)
    phases = {
        'Phase 1: General Collections': {
            'start': start_date,
            'duration': 26,  # 6 months
            'color': '#3b82f6',
            'deliverables': ['System Setup', 'Staff Training', 'Quality Validation', 'Process Documentation']
        },
        'Phase 2: Special Collections': {
            'start': start_date + timedelta(weeks=26),
            'duration': 26,  # 6 months
            'color': '#f59e0b', 
            'deliverables': ['Domain Classification', 'Historical Analysis', 'Specialist Training', 'Integration Testing']
        },
        'Phase 3: Real-Time Integration': {
            'start': start_date + timedelta(weeks=52),
            'duration': 26,  # 6 months
            'color': '#10b981',
            'deliverables': ['Alma Integration', 'Workflow Automation', 'Performance Optimization', 'Full Deployment']
        }
    }
    
    # 1. Gantt Chart Style Timeline
    y_pos = 0
    for phase_name, details in phases.items():
        # Draw phase bar
        ax1.barh(y_pos, details['duration'], left=0, height=0.6, 
                color=details['color'], alpha=0.7, edgecolor='white', linewidth=2)
        
        # Add phase label
        ax1.text(-2, y_pos, phase_name, ha='right', va='center', 
                fontweight='bold', fontsize=11)
        
        # Add duration text
        ax1.text(details['duration']/2, y_pos, f"{details['duration']} weeks", 
                ha='center', va='center', color='white', fontweight='bold')
        
        y_pos += 1
    
    # Add milestones and decision points
    milestone_weeks = [13, 26, 39, 52, 65, 78]
    milestone_labels = ['Phase 1 Midpoint', 'Decision Point 1', 'Phase 2 Midpoint', 
                       'Decision Point 2', 'Phase 3 Midpoint', 'Full Production']
    
    for week, label in zip(milestone_weeks, milestone_labels):
        ax1.axvline(x=week, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(week, 3.2, label, rotation=45, ha='left', va='bottom', 
                fontsize=9, color='red', fontweight='bold')
    
    ax1.set_xlim(-15, 85)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_xlabel('Timeline (Weeks from Start)', fontsize=12, fontweight='bold')
    ax1.set_title('Implementation Timeline: Phased Deployment Strategy', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Remove y-axis ticks since we have manual labels
    ax1.set_yticks([])
    
    # 2. Resource Allocation Timeline
    weeks = np.arange(0, 79)
    
    # Resource allocation by phase (FTE by week)
    setup_resources = np.zeros(78)
    setup_resources[0:4] = 0.5  # Initial setup
    
    phase1_resources = np.zeros(78)
    phase1_resources[4:26] = 1.0  # Full implementation
    
    phase2_resources = np.zeros(78)  
    phase2_resources[26:52] = 1.2  # Increased complexity
    
    phase3_resources = np.zeros(78)
    phase3_resources[52:78] = 0.8  # Optimization and integration
    
    # Stack the resource allocation
    ax2.fill_between(weeks, 0, setup_resources, alpha=0.7, color='#6b7280', label='Setup & Planning')
    ax2.fill_between(weeks, setup_resources, setup_resources + phase1_resources, 
                    alpha=0.7, color='#3b82f6', label='Phase 1: General Collections')
    ax2.fill_between(weeks, setup_resources + phase1_resources, 
                    setup_resources + phase1_resources + phase2_resources,
                    alpha=0.7, color='#f59e0b', label='Phase 2: Special Collections')
    ax2.fill_between(weeks, setup_resources + phase1_resources + phase2_resources,
                    setup_resources + phase1_resources + phase2_resources + phase3_resources,
                    alpha=0.7, color='#10b981', label='Phase 3: Real-Time Integration')
    
    # Add decision points
    for week in [26, 52]:
        ax2.axvline(x=week, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Timeline (Weeks from Start)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resource Allocation (FTE)', fontsize=12, fontweight='bold')
    ax2.set_title('Resource Allocation Over Implementation Timeline', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 78)
    
    plt.tight_layout()
    
    return fig

def create_success_metrics_dashboard():
    """
    Creates a dashboard showing how we'll measure success at each phase,
    providing concrete evidence for decision points.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Phase 1 Success Metrics
    phase1_metrics = ['Precision\nMaintained', 'Staff\nConfidence', 'Quality\nImprovements', 
                     'Process\nEfficiency', 'Error\nReduction']
    phase1_targets = [99, 80, 75, 60, 85]
    phase1_current = [99.2, 85, 82, 68, 88]  # Simulated actual results
    
    x_pos = np.arange(len(phase1_metrics))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, phase1_targets, width, label='Target', 
                   color='#94a3b8', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, phase1_current, width, label='Achieved', 
                   color='#10b981', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Success Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Phase 1: General Collections Results', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(phase1_metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Quality Trend Analysis
    weeks = np.arange(1, 27)  # 26 weeks of Phase 1
    
    # Simulated quality metrics over time
    precision_trend = 99.55 + np.random.normal(0, 0.1, 26)  # Very stable
    confidence_trend = 60 + 25 * (1 - np.exp(-weeks/8))  # Learning curve
    efficiency_trend = 30 + 40 * (1 - np.exp(-weeks/10))  # Gradual improvement
    
    ax2.plot(weeks, precision_trend, 'g-', linewidth=3, label='Precision (%)', marker='o', markersize=4)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(weeks, confidence_trend, 'b-', linewidth=3, label='Staff Confidence (%)', marker='s', markersize=4)
    ax2_twin.plot(weeks, efficiency_trend, 'orange', linewidth=3, label='Process Efficiency (%)', marker='^', markersize=4)
    
    ax2.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision (%)', fontsize=12, fontweight='bold', color='green')
    ax2_twin.set_ylabel('Confidence & Efficiency (%)', fontsize=12, fontweight='bold', color='blue')
    ax2.set_title('Phase 1: Quality Trends Over Time', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    ax2.grid(True, alpha=0.3)
    
    # 3. Decision Point Criteria
    ax3.axis('off')
    
    decision_criteria = """
DECISION POINT CRITERIA

✅ Phase 1 → Phase 2 (Month 6):
   • Precision ≥ 99% maintained
   • Staff confidence ≥ 80%
   • Quality improvements ≥ 70%
   • No critical system issues
   • Training completion rate ≥ 95%

✅ Phase 2 → Phase 3 (Month 12):
   • Special collections integration successful
   • Domain classification accuracy ≥ 85%
   • Specialist staff adoption ≥ 80%
   • Historical disambiguation validated
   • Performance metrics stable

✅ Phase 3 → Production (Month 18):
   • Real-time integration functional
   • Alma workflow optimization complete
   • Full automation features tested
   • Staff productivity gains documented
   • Long-term sustainability confirmed

🛑 Stop/Reassess Criteria:
   • Precision drops below 95%
   • Staff resistance >30%
   • Critical quality failures
   • Cost overrun >50%
   • Timeline delay >3 months
"""
    
    ax3.text(0.05, 0.95, decision_criteria, transform=ax3.transAxes, fontsize=10,
            va='top', ha='left', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='#f0f9ff', alpha=0.8))
    
    # 4. Cumulative Value Creation
    months = np.arange(1, 19)  # 18-month implementation
    
    # Value creation by category (cumulative)
    quality_value = np.cumsum([2000] * 6 + [4000] * 6 + [6000] * 6)  # Increasing value
    efficiency_value = np.cumsum([1000] * 6 + [3000] * 6 + [5000] * 6)
    cost_avoidance = np.cumsum([3000] * 6 + [5000] * 6 + [7000] * 6)
    
    ax4.fill_between(months, 0, quality_value, alpha=0.7, color='#3b82f6', label='Quality Improvements')
    ax4.fill_between(months, quality_value, quality_value + efficiency_value, 
                    alpha=0.7, color='#10b981', label='Efficiency Gains')
    ax4.fill_between(months, quality_value + efficiency_value, 
                    quality_value + efficiency_value + cost_avoidance,
                    alpha=0.7, color='#f59e0b', label='Cost Avoidance')
    
    # Add phase boundaries
    ax4.axvline(x=6, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax4.axvline(x=12, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Value ($)', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Value Creation by Phase', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Success Metrics and Decision Points Dashboard', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# Generate and save the visualizations
if __name__ == "__main__":
    # Create risk assessment matrix
    fig1 = create_risk_assessment_matrix()
    fig1.savefig('risk_assessment_matrix.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    # Create implementation timeline
    fig2 = create_implementation_timeline()
    fig2.savefig('implementation_timeline.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    # Create success metrics dashboard
    fig3 = create_success_metrics_dashboard()
    fig3.savefig('success_metrics_dashboard.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    print("Risk and implementation visualizations saved as:")
    print("- risk_assessment_matrix.png")
    print("- implementation_timeline.png")
    print("- success_metrics_dashboard.png")
    
    plt.show()
