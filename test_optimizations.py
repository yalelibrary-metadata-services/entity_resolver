#!/usr/bin/env python3
"""
Test script to validate the performance optimizations for embedding and indexing.

This script performs a small-scale test to ensure the optimizations work correctly
before running the full 10-day processing pipeline.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config_utils import load_config_with_environment
from src.embedding_and_indexing import EmbeddingAndIndexingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(size: int = 100) -> Dict[str, Any]:
    """
    Create synthetic test data to validate the pipeline.
    
    Args:
        size: Number of test strings to create
        
    Returns:
        Dictionary with test data structures
    """
    # Create synthetic test data
    string_dict = {}
    field_hash_mapping = {}
    string_counts = {}
    
    for i in range(size):
        hash_val = f"test_hash_{i:04d}"
        string_val = f"Test entity string number {i} with various content to embed"
        
        string_dict[hash_val] = string_val
        field_hash_mapping[hash_val] = {"composite": 1, "person": 1}
        string_counts[hash_val] = 1
    
    logger.info(f"Created {size} synthetic test records")
    return {
        'string_dict': string_dict,
        'field_hash_mapping': field_hash_mapping,
        'string_counts': string_counts
    }

def test_optimizations(config_path: str = 'config.yml', test_size: int = 50):
    """
    Test the optimized embedding and indexing pipeline.
    
    Args:
        config_path: Path to configuration file
        test_size: Number of test records to process
    """
    logger.info("="*60)
    logger.info("TESTING OPTIMIZED EMBEDDING AND INDEXING PIPELINE")
    logger.info("="*60)
    
    try:
        # Load configuration
        config = load_config_with_environment(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Log current optimization settings
        logger.info("OPTIMIZATION SETTINGS:")
        logger.info(f"  - Batch size: {config.get('embedding_batch_size', 'not set')}")
        logger.info(f"  - Max tokens/minute: {config.get('max_tokens_per_minute', 'not set'):,}")
        logger.info(f"  - Max requests/minute: {config.get('max_requests_per_minute', 'not set'):,}")
        logger.info(f"  - Workers: {config.get('embedding_workers', 'not set')}")
        logger.info(f"  - Daily polling interval: {config.get('tpd_poll_interval', 'not set')} seconds")
        
        # Create test data
        test_data = create_test_data(test_size)
        
        # Initialize pipeline
        logger.info(f"Initializing pipeline for {test_size} test records...")
        pipeline = EmbeddingAndIndexingPipeline(config)
        
        # Test directory
        test_checkpoint_dir = "data/checkpoints/test"
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        
        # Run the test
        start_time = time.time()
        
        logger.info("Starting test processing...")
        metrics = pipeline.process(
            test_data['string_dict'],
            test_data['field_hash_mapping'], 
            test_data['string_counts'],
            test_checkpoint_dir
        )
        
        elapsed_time = time.time() - start_time
        
        # Calculate performance metrics
        throughput = test_size / elapsed_time if elapsed_time > 0 else 0
        
        logger.info("="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Records processed: {metrics.get('strings_processed', 0)}/{test_size}")
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} records/second")
        logger.info(f"Tokens used: {metrics.get('tokens_used', 0):,}")
        logger.info(f"Status: {metrics.get('status', 'unknown')}")
        
        # Estimate full processing time
        if throughput > 0:
            # Assume you have approximately 1M records to process (adjust as needed)
            estimated_records = 1_000_000  # Adjust this based on your actual dataset size
            estimated_time_hours = estimated_records / throughput / 3600
            estimated_time_days = estimated_time_hours / 24
            
            logger.info("="*60)
            logger.info("FULL PIPELINE ESTIMATES")
            logger.info("="*60)
            logger.info(f"Estimated records to process: {estimated_records:,}")
            logger.info(f"Estimated processing time: {estimated_time_hours:.1f} hours ({estimated_time_days:.1f} days)")
            
            if estimated_time_days < 3:
                logger.info("🎉 EXCELLENT: Processing should complete in under 3 days!")
            elif estimated_time_days < 6:
                logger.info("✅ GOOD: Significant improvement achieved!")
            elif estimated_time_days < 8:
                logger.info("🆗 MODERATE: Some improvement, but more optimization possible")
            else:
                logger.info("⚠️  LIMITED IMPACT: Your rate limits may already be optimized")
        
        # Print rate limit status
        status = pipeline.get_processing_status()
        daily_usage = status['daily_token_usage']
        logger.info("="*60)
        logger.info("RATE LIMIT STATUS")
        logger.info("="*60)
        logger.info(f"Daily token usage: {daily_usage['tokens_used']:,}/{daily_usage['tokens_limit']:,} ({daily_usage['usage_percentage']:.1f}%)")
        
        if pipeline.failed_requests:
            logger.warning(f"Failed requests: {len(pipeline.failed_requests)}")
        else:
            logger.info("✅ No failed requests")
            
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False
        
    finally:
        # Cleanup
        if 'pipeline' in locals() and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
                logger.info("Weaviate client connection closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {str(e)}")

def main():
    """Main function to run the optimization test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test optimized embedding and indexing')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--size', type=int, default=50, help='Number of test records to process')
    args = parser.parse_args()
    
    success = test_optimizations(args.config, args.size)
    
    if success:
        logger.info("✅ Test completed successfully!")
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS:")
        logger.info("1. If test shows <3 days estimated time: Run your full pipeline!")
        logger.info("2. If test shows 3-6 days: Good improvement achieved, consider running")
        logger.info("3. If test shows >6 days: Limited improvement due to already-optimized rate limits")
        logger.info("4. Consider OpenAI Batch API for 50% cost savings (24hr turnaround)")
        logger.info("="*60)
    else:
        logger.error("❌ Test failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()