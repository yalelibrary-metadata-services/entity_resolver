#!/usr/bin/env python3
"""
Batch Manager for Entity Resolution

This script provides manual management of OpenAI batch jobs for embedding generation.
Use this when you want to create batch jobs and manually check their status rather
than having the script continuously poll for completion.
"""

import os
import sys
import logging
import argparse
import yaml
import pickle
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embedding_and_indexing_batch import BatchEmbeddingPipeline

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def load_preprocessing_data(checkpoint_dir: str) -> tuple:
    """Load preprocessing data from checkpoints."""
    try:
        # Load string dictionary
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        # Load field hash mapping
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        # Load string counts
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        return string_dict, field_hash_mapping, string_counts
        
    except FileNotFoundError as e:
        print(f"Preprocessing data not found: {str(e)}")
        print("Run preprocessing stage first: python main.py --config config.yml --start preprocessing --end preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading preprocessing data: {str(e)}")
        sys.exit(1)

def create_jobs(config: Dict[str, Any]) -> None:
    """Create batch jobs without waiting for completion."""
    print("🚀 Creating OpenAI batch jobs...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Load preprocessing data
    string_dict, field_hash_mapping, string_counts = load_preprocessing_data(checkpoint_dir)
    
    print(f"📊 Loaded preprocessing data: {len(string_dict)} strings")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Create batch jobs
        result = pipeline.create_batch_jobs_only(
            string_dict, field_hash_mapping, string_counts, checkpoint_dir
        )
        
        if result['status'] == 'jobs_created':
            print(f"\n✅ Successfully created {result['jobs_created']} batch jobs!")
            print(f"📋 Total requests: {result['total_requests']:,}")
            print(f"💰 Estimated cost savings: ${result['estimated_cost_savings']:.4f}")
            print(f"⏱️  Creation time: {result['elapsed_time']:.2f} seconds")
            print(f"\n📋 Next steps:")
            print(f"   1. Wait for OpenAI to process your jobs (up to 24 hours)")
            print(f"   2. Check status: python batch_manager.py --status")
            print(f"   3. Download results: python batch_manager.py --download")
        elif result['status'] == 'no_work':
            print("ℹ️  No new work to process - all eligible strings already processed")
        else:
            print(f"❌ Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Error creating batch jobs: {str(e)}")
        sys.exit(1)
    finally:
        if pipeline and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
            except:
                pass

def check_status(config: Dict[str, Any]) -> None:
    """Check status of batch jobs."""
    print("📊 Checking batch job status...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Check status
        result = pipeline.check_batch_status(checkpoint_dir)
        
        if result['status'] == 'no_jobs':
            print("ℹ️  No batch jobs found.")
            print("   Create jobs first: python batch_manager.py --create")
        elif result['status'] == 'checked':
            # Status already logged by the method
            if result['ready_for_download']:
                print(f"\n🎉 {result['completed']} jobs are ready for download!")
                print(f"   Run: python batch_manager.py --download")
            elif result['pending'] > 0 or result['in_progress'] > 0:
                print(f"\n⏳ Jobs are still processing. Check again later.")
            else:
                print(f"\n⚠️  All jobs have completed or failed. Check logs for details.")
        else:
            print(f"❌ Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Error checking batch status: {str(e)}")
        sys.exit(1)
    finally:
        if pipeline and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
            except:
                pass

def download_results(config: Dict[str, Any]) -> None:
    """Download and process completed batch results."""
    print("📥 Processing completed batch jobs...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Process completed jobs
        result = pipeline.process_completed_jobs(checkpoint_dir)
        
        if result['status'] == 'no_jobs':
            print("ℹ️  No batch jobs found.")
            print("   Create jobs first: python batch_manager.py --create")
        elif result['status'] == 'no_completed_jobs':
            print("ℹ️  No completed jobs found.")
            print("   Check status first: python batch_manager.py --status")
        elif result['status'] == 'completed':
            # Results already logged by the method
            print(f"\n🎉 Batch processing completed successfully!")
            if result['collection_count']:
                print(f"   Weaviate collection now contains {result['collection_count']:,} objects")
        else:
            print(f"❌ Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Error processing batch results: {str(e)}")
        sys.exit(1)
    finally:
        if pipeline and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
            except:
                pass

def investigate_failures(config: Dict[str, Any]) -> None:
    """Investigate failed batch jobs to understand why they failed."""
    print("🔍 Investigating failed batch jobs...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Load or recover batch jobs
        pipeline.load_checkpoint(checkpoint_dir)
        if not pipeline.batch_jobs:
            pipeline._recover_batch_jobs_from_api()
        
        if not pipeline.batch_jobs:
            print("ℹ️  No batch jobs found to investigate")
            return
        
        failed_jobs = []
        for job_id, job_info in pipeline.batch_jobs.items():
            if job_info.get('status') == 'failed':
                failed_jobs.append(job_id)
        
        if not failed_jobs:
            print("ℹ️  No failed jobs found")
            return
        
        print(f"🔍 Investigating {len(failed_jobs)} failed jobs...")
        
        # Sample a few failed jobs to investigate
        sample_size = min(5, len(failed_jobs))
        sample_jobs = failed_jobs[:sample_size]
        
        for i, job_id in enumerate(sample_jobs):
            print(f"\n--- Failed Job {i+1}/{sample_size}: {job_id} ---")
            
            try:
                batch_status = pipeline.openai_client.batches.retrieve(job_id)
                
                print(f"Status: {batch_status.status}")
                print(f"Created: {batch_status.created_at}")
                
                if hasattr(batch_status, 'request_counts') and batch_status.request_counts:
                    counts = batch_status.request_counts
                    print(f"Requests - Total: {getattr(counts, 'total', 0)}, "
                          f"Completed: {getattr(counts, 'completed', 0)}, "
                          f"Failed: {getattr(counts, 'failed', 0)}")
                
                if hasattr(batch_status, 'errors') and batch_status.errors:
                    print(f"Errors: {batch_status.errors}")
                
                # Try to get error file if available
                if hasattr(batch_status, 'error_file_id') and batch_status.error_file_id:
                    print(f"Error file ID: {batch_status.error_file_id}")
                    try:
                        error_content = pipeline.openai_client.files.content(batch_status.error_file_id)
                        error_lines = error_content.content.decode('utf-8').split('\n')[:5]  # First 5 error lines
                        print("Sample errors:")
                        for line in error_lines:
                            if line.strip():
                                print(f"  {line}")
                    except Exception as e:
                        print(f"  Could not retrieve error details: {e}")
                
            except Exception as e:
                print(f"Error investigating job {job_id}: {e}")
        
        print(f"\n📊 Summary:")
        print(f"   Total jobs: {len(pipeline.batch_jobs)}")
        print(f"   Failed jobs: {len(failed_jobs)}")
        print(f"   Failure rate: {len(failed_jobs)/len(pipeline.batch_jobs)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error investigating failures: {str(e)}")
        sys.exit(1)
    finally:
        if pipeline and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
            except:
                pass

def recover_all_jobs(config: Dict[str, Any]) -> None:
    """Recover all batch jobs from OpenAI API."""
    print("🔄 Recovering all batch jobs from OpenAI API...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Force recovery from API (clear local batch jobs first)
        pipeline.batch_jobs = {}
        pipeline._recover_batch_jobs_from_api()
        
        # Save the recovered jobs
        pipeline.save_checkpoint(checkpoint_dir)
        
        if pipeline.batch_jobs:
            print(f"\n🎉 Successfully recovered {len(pipeline.batch_jobs)} batch jobs!")
            
            # Group by status for summary
            status_counts = {}
            for job_info in pipeline.batch_jobs.values():
                status = job_info['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\n📊 Recovered job status summary:")
            for status, count in status_counts.items():
                print(f"   {status}: {count} jobs")
                
            print(f"\n📋 Next steps:")
            print(f"   1. Check status: python batch_manager.py --status")
            print(f"   2. Download completed results: python batch_manager.py --download")
            print(f"   3. Create new jobs if needed: python batch_manager.py --create")
        else:
            print("ℹ️  No entity resolution batch jobs found in OpenAI API")
            
    except Exception as e:
        print(f"❌ Error recovering batch jobs: {str(e)}")
        sys.exit(1)
    finally:
        if pipeline and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
            except:
                pass

def reset_embedding_stage(config: Dict[str, Any]) -> None:
    """Reset all embedding_and_indexing stage data."""
    print("🗑️  Resetting embedding_and_indexing stage...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    files_deleted = []
    
    try:
        # Delete real-time processing checkpoints
        real_time_files = [
            os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        ]
        
        # Delete batch processing checkpoints
        batch_files = [
            os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl'),
            os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        ]
        
        # Delete batch request/result files
        import glob
        batch_request_pattern = os.path.join(checkpoint_dir, 'batch_requests_*.jsonl')
        batch_result_pattern = os.path.join(checkpoint_dir, 'batch_results_*.jsonl')
        
        batch_files.extend(glob.glob(batch_request_pattern))
        batch_files.extend(glob.glob(batch_result_pattern))
        
        # Combine all files to delete
        all_files = real_time_files + batch_files
        
        # Delete files
        for file_path in all_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                files_deleted.append(os.path.basename(file_path))
                print(f"   ✅ Deleted: {os.path.basename(file_path)}")
        
        if files_deleted:
            print(f"\n🎉 Successfully deleted {len(files_deleted)} files:")
            for filename in files_deleted:
                print(f"   • {filename}")
        else:
            print("ℹ️  No files found to delete - stage already clean")
        
        print(f"\n📋 Embedding stage reset complete. You can now:")
        print(f"   • Run embedding with real-time processing")
        print(f"   • Run embedding with batch processing: python batch_manager.py --create")
        
    except Exception as e:
        print(f"❌ Error resetting embedding stage: {str(e)}")
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Manual Batch Manager for Entity Resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create batch jobs
  python batch_manager.py --create
  
  # Check job status
  python batch_manager.py --status
  
  # Download and process results
  python batch_manager.py --download
  
  # Recover all batch jobs from OpenAI (if checkpoints corrupted)
  python batch_manager.py --recover
  
  # Reset embedding stage (clear all checkpoints)
  python batch_manager.py --reset
  
  # Full workflow
  python batch_manager.py --create
  # ... wait for jobs to complete ...
  python batch_manager.py --status
  python batch_manager.py --download
        """
    )
    
    parser.add_argument('--config', default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    
    # Action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--create', action='store_true',
                             help='Create batch jobs and upload to OpenAI')
    action_group.add_argument('--status', action='store_true',
                             help='Check status of existing batch jobs')
    action_group.add_argument('--download', action='store_true',
                             help='Download and process completed batch results')
    action_group.add_argument('--reset', action='store_true',
                             help='Reset embedding_and_indexing stage (clear all checkpoints and files)')
    action_group.add_argument('--recover', action='store_true',
                             help='Recover all batch jobs from OpenAI API (useful when checkpoints are corrupted)')
    action_group.add_argument('--investigate', action='store_true',
                             help='Investigate failed batch jobs to understand failure reasons')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config(args.config)
    
    # For reset, recover, and investigate commands, we don't need batch processing to be enabled
    if not args.reset and not args.recover and not args.investigate and not config.get('use_batch_embeddings', False):
        print("⚠️  Batch embeddings are not enabled in configuration.")
        print("   Set 'use_batch_embeddings: true' in your config.yml")
        sys.exit(1)
    
    # Execute requested action
    try:
        if args.create:
            create_jobs(config)
        elif args.status:
            check_status(config)
        elif args.download:
            download_results(config)
        elif args.reset:
            reset_embedding_stage(config)
        elif args.recover:
            recover_all_jobs(config)
        elif args.investigate:
            investigate_failures(config)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()