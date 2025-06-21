#!/usr/bin/env python3
"""
Entity Resolution Pipeline for Yale University Library Catalog

Main entry point for running the entity resolution pipeline.
This script coordinates the execution of all pipeline stages.

Usage:
  python main.py --config config.yml [--start STAGE] [--end STAGE] [--reset STAGE1 STAGE2 ...]
"""

import os
import sys
import logging
import argparse
import yaml
import time
from typing import Dict, List, Any

# Add the project root and src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline modules
from src.utils import setup_logging, create_directory_structure, get_memory_usage
from src.orchestrating import PipelineOrchestrator

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    
    parser.add_argument('--config', default='config.yml',
                      help='Path to configuration file')
    
    parser.add_argument('--start',
                      help='Stage to start from (preprocessing, embedding_and_indexing, training, classifying, reporting)')
    
    parser.add_argument('--end',
                      help='Stage to end at (preprocessing, embedding_and_indexing, training, classifying, reporting)')
    
    parser.add_argument('--reset', nargs='*',
                      help='Stages to reset before running')
    
    parser.add_argument('--docker', action='store_true',
                      help='Run Weaviate in Docker')
    
    parser.add_argument('--resume', action='store_true',
                      help='Resume pipeline from last completed stage')
    
    parser.add_argument('--status', action='store_true',
                      help='Show current pipeline execution status and exit')
                      
    parser.add_argument('--disable-scaling', action='store_true',
                      help='EXPERIMENTAL: Disable all feature scaling and use raw values directly')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def start_weaviate_docker(config):
    """
    Start Weaviate using Docker Compose.
    
    Args:
        config: Configuration dictionary
    """
    try:
        import subprocess
        docker_compose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docker-compose.yml')
        
        if not os.path.exists(docker_compose_path):
            print(f"Docker Compose file not found: {docker_compose_path}")
            sys.exit(1)
        
        print("Starting Weaviate using Docker Compose...")
        subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
        
        # Wait for Weaviate to be ready
        print("Waiting for Weaviate to be ready...")
        time.sleep(10)
        
    except Exception as e:
        print(f"Error starting Weaviate Docker: {str(e)}")
        sys.exit(1)

def main():
    """
    Main function to run the entity resolution pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create directory structure
    create_directory_structure(config)
    
    # Start Weaviate Docker if requested
    if args.docker:
        start_weaviate_docker(config)
    
    # Initialize checkpointing system
    from src.checkpoint_manager import get_checkpoint_manager
    checkpoint_manager = get_checkpoint_manager(config)
    
    # Display status if requested
    if args.status:
        progress = checkpoint_manager.get_pipeline_progress()
        print("\nEntity Resolution Pipeline Status")
        print("===================================\n")
        print(f"Last completed stage: {progress['last_completed_stage'] or 'None'}")
        print(f"Completed stages: {', '.join(progress['completed_stages']) if progress['completed_stages'] else 'None'}")
        
        # Determine next stage to run
        orchestrator = PipelineOrchestrator(args.config)
        next_stage = checkpoint_manager.get_next_stage(orchestrator.stages)
        print(f"Next stage to run: {next_stage or 'All stages completed'}\n")
        
        return
    
    # Log initial information
    logger.info("Starting Entity Resolution Pipeline")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Initial memory usage: {get_memory_usage()}")
    
    # Add disable_scaling flag to config if specified
    if args.disable_scaling:
        config['disable_scaling'] = True
        logger.warning("*** EXPERIMENTAL: Feature scaling completely disabled - will use raw feature values ***")
    
    # Analyze feature configuration
    from src.utils import analyze_feature_configuration
    feature_analysis = analyze_feature_configuration(config)
    logger.info(f"Enabled base features: {feature_analysis['enabled_features']}")
    logger.info(f"Enabled custom features: {feature_analysis['custom_features']}")
    logger.info(f"Active feature substitutions: {feature_analysis['substitutions']}")
    logger.info(f"Effective feature set: {feature_analysis['effective_features']}")
    
    
    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(args.config)
    
    # Set recreate_collections flag if embedding_and_indexing is being reset
    if args.reset and 'embedding_and_indexing' in args.reset:
        orchestrator.config['recreate_collections'] = True
        logger.info("Setting recreate_collections flag to True due to reset flag")
        
    try:
        # Run pipeline
        metrics = orchestrator.run_pipeline(
            args.start, 
            args.end, 
            args.reset,
            args.resume  # Pass resume flag
        )
        
        # Log completion
        logger.info("Entity Resolution Pipeline completed successfully")
        logger.info(f"Total runtime: {metrics['total_runtime']:.2f} seconds")
        logger.info(f"Final memory usage: {get_memory_usage()}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
