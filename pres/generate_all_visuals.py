#!/usr/bin/env python3
"""
Master Visualization Generator for Yale AI Workshop
Generates all graphics needed for the slideshow presentation
"""

import sys
import os
import subprocess
import traceback
from pathlib import Path

def setup_paths():
    """Setup paths for script execution"""
    # Get the current directory (should be pres/)
    current_dir = Path(__file__).parent
    img_dir = current_dir / "img"
    
    # Create img directory if it doesn't exist
    img_dir.mkdir(exist_ok=True)
    
    return current_dir, img_dir

def run_visualization_script(script_path, description):
    """Run a visualization script and handle errors"""
    print(f"\n🎨 Generating: {description}")
    print(f"📁 Running: {script_path}")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                               capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Error in {description}:")
            print(f"   {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {description} took too long")
        return False
    except Exception as e:
        print(f"💥 Exception in {description}: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Generate all visualizations for the workshop"""
    print("🚀 Yale AI Workshop - Master Visualization Generator")
    print("=" * 60)
    
    # Setup paths
    current_dir, img_dir = setup_paths()
    print(f"📂 Working directory: {current_dir}")
    print(f"🖼️  Output directory: {img_dir}")
    
    # List of all visualization scripts to run
    visualization_scripts = [
        ("viz_journey_timeline.py", "Journey Timeline - Evolution from embeddings to production"),
        ("viz_franz_schubert_tree.py", "Franz Schubert Decision Tree - Disambiguation flowchart"),
        ("viz_tokenization_analysis.py", "Tokenization Analysis - Multi-language comparison"),
        ("viz_setfit_mistral_comparison.py", "SetFit vs Mistral - Capability comparison matrix"),
        ("viz_similarity_heatmaps.py", "Similarity Heatmaps - Vector similarity visualization"),
        ("viz_threshold_problem.py", "Threshold Problem - Why single thresholds fail"),
        ("viz_feature_importance.py", "Feature Importance - ML feature weights radar chart"),
        ("viz_feature_radar_simple.py", "Feature Radar - Clean simplified radar chart"),
        ("viz_feature_weights_bar.py", "Feature Weights - Clean bar chart with directions"),
        ("viz_feature_descriptions.py", "Feature Descriptions - Individual feature explanation cards"),
        ("viz_schubert_feature_analysis.py", "Schubert Analysis - Detailed feature breakdown example"),
        ("viz_production_metrics.py", "Production Metrics - Yale system performance dashboard"),
        ("viz_cost_benefit.py", "Cost-Benefit Analysis - Manual vs automated comparison"),
        ("viz_pipeline_architecture.py", "Pipeline Architecture - Complete system diagram"),
        ("viz_weaviate_workflow.py", "Weaviate Workflow - Vector database integration"),
        ("viz_hotdeck_imputation.py", "Hot-deck Imputation - Missing data imputation process"),
        ("viz_embedding_evolution.py", "Embedding Evolution - Word2Vec to transformers timeline"),
        ("viz_taxonomy_tree.py", "Taxonomy Tree - Yale domain classification hierarchy"),
        ("viz_workshop_overview.py", "Workshop Overview - Learning objectives summary")
    ]
    
    # Track results
    total_scripts = len(visualization_scripts)
    successful = 0
    failed = []
    
    print(f"\n📋 Generating {total_scripts} visualizations...")
    
    # Run each visualization script
    for script_name, description in visualization_scripts:
        script_path = current_dir / script_name
        
        if not script_path.exists():
            print(f"⚠️  Script not found: {script_name}")
            failed.append((script_name, "Script file not found"))
            continue
            
        if run_visualization_script(script_path, description):
            successful += 1
        else:
            failed.append((script_name, "Execution failed"))
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 VISUALIZATION GENERATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{total_scripts}")
    print(f"❌ Failed: {len(failed)}/{total_scripts}")
    
    if failed:
        print(f"\n💥 Failed Scripts:")
        for script_name, reason in failed:
            print(f"   • {script_name}: {reason}")
    
    # Check generated files
    print(f"\n📁 Generated Files in {img_dir}:")
    image_files = list(img_dir.glob("*.png"))
    for img_file in sorted(image_files):
        file_size = img_file.stat().st_size / 1024  # KB
        print(f"   📷 {img_file.name} ({file_size:.1f} KB)")
    
    print(f"\n📈 Total images generated: {len(image_files)}")
    
    # Final status
    if successful == total_scripts:
        print("\n🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("🎯 Workshop slideshow is ready to present.")
        return 0
    else:
        print(f"\n⚠️  {len(failed)} visualizations failed to generate.")
        print("   You may need to fix these scripts or create manual alternatives.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)