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
import fcntl
import time
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embedding_and_indexing_batch import BatchEmbeddingPipeline

class ProcessLock:
    """Simple file-based process lock to prevent multiple instances."""
    
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.lock_fd = None
    
    def __enter__(self):
        """Acquire the lock."""
        try:
            # Create lock directory if it doesn't exist
            os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
            
            # Open lock file
            self.lock_fd = open(self.lock_file, 'w')
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write our process ID to the lock file
            self.lock_fd.write(f"{os.getpid()}\n{time.time()}\n")
            self.lock_fd.flush()
            os.fsync(self.lock_fd.fileno())  # Force write to disk
            
            return self
            
        except (IOError, OSError) as e:
            # Lock is already held by another process
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            
            # Check if this is a "resource unavailable" error (lock already held)
            if e.errno == 35 or e.errno == 11 or "Resource temporarily unavailable" in str(e):
                # Try to read the existing lock file to show which process holds it
                try:
                    with open(self.lock_file, 'r') as f:
                        content = f.read().strip()
                        lines = content.split('\n') if content else []
                        if len(lines) >= 2 and lines[0] and lines[1]:
                            try:
                                pid = lines[0]
                                timestamp = float(lines[1])
                                lock_age = time.time() - timestamp
                                print(f"❌ Another batch creation process is already running!")
                                print(f"   Process ID: {pid}")
                                print(f"   Lock age: {lock_age:.1f} seconds")
                                print(f"   Lock file: {self.lock_file}")
                                print(f"\n💡 Options:")
                                print(f"   1. Wait for the other process to complete")
                                print(f"   2. Kill the other process: kill {pid}")
                                print(f"   3. Remove stale lock (if process is dead): rm {self.lock_file}")
                            except (ValueError, IndexError):
                                print(f"❌ Another batch creation process is running (lock file: {self.lock_file})")
                        else:
                            print(f"❌ Another batch creation process is running (lock file: {self.lock_file})")
                except:
                    print(f"❌ Another batch creation process is running (lock file: {self.lock_file})")
            else:
                # Some other I/O error
                print(f"❌ Error acquiring lock: {e}")
            
            raise SystemExit(1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                # Remove lock file
                try:
                    os.unlink(self.lock_file)
                except:
                    pass
            except:
                pass
            finally:
                self.lock_fd = None

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
    lock_file = os.path.join(checkpoint_dir, ".batch_creation.lock")
    
    # Use process lock to prevent multiple creation processes
    with ProcessLock(lock_file):
        print("🔒 Acquired process lock - proceeding with batch creation")
        
        # Load preprocessing data
        string_dict, field_hash_mapping, string_counts = load_preprocessing_data(checkpoint_dir)
        
        print(f"📊 Loaded preprocessing data: {len(string_dict)} strings")
        
        # Initialize batch pipeline with proper context management
        try:
            with BatchEmbeddingPipeline(config) as pipeline:
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
        
        print("🔓 Released process lock")

def check_status(config: Dict[str, Any]) -> None:
    """Check status of batch jobs."""
    print("📊 Checking batch job status...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Debug: List existing files in checkpoint directory
    print(f"🔍 Debug: Checking checkpoint directory: {checkpoint_dir}")
    if os.path.exists(checkpoint_dir):
        import glob
        jsonl_files = glob.glob(os.path.join(checkpoint_dir, "*.jsonl"))
        if jsonl_files:
            print(f"   Found {len(jsonl_files)} JSONL files:")
            for file in jsonl_files:
                print(f"     - {os.path.basename(file)}")
        else:
            print("   No JSONL files found in checkpoint directory")
    else:
        print(f"   Checkpoint directory does not exist: {checkpoint_dir}")
    
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
            # Status already logged by the method - now show additional details
            
            # Show granular status breakdown if available
            if 'status_counts' in result:
                status_counts = result['status_counts']
                print(f"\n🔍 GRANULAR STATUS BREAKDOWN:")
                
                # Define status display order and emojis
                status_display = [
                    ('pending', "⏳ Pending"),
                    ('validating', "🔍 Validating"), 
                    ('in_progress', "🔄 In Progress"),
                    ('finalizing', "🏁 Finalizing"),
                    ('completed', "✅ Completed"),
                    ('failed', "❌ Failed"),
                    ('expired', "⏰ Expired"),
                    ('cancelled', "🚫 Cancelled"),
                    ('error', "⚠️  Error")
                ]
                
                # Show each status if count > 0
                for status_key, label in status_display:
                    count = status_counts.get(status_key, 0)
                    if count > 0:
                        print(f"   {label}: {count}")
            
            if result['ready_for_download']:
                print(f"\n🎉 {result['completed']} jobs are ready for download!")
                print(f"   Run: python batch_manager.py --download")
                
                # Show detailed download/processing status for completed jobs
                job_statuses = result.get('job_statuses', {})
                completed_jobs = [job for job in job_statuses.values() if job['status'] == 'completed']
                
                if completed_jobs:
                    downloaded_count = sum(1 for job in completed_jobs if job.get('download_status') == 'downloaded')
                    processed_count = sum(1 for job in completed_jobs if job.get('processing_status') == 'processed')
                    
                    print(f"\n📊 Download/Processing Status:")
                    print(f"   ✅ Completed: {len(completed_jobs)} jobs")
                    print(f"   📥 Downloaded: {downloaded_count} jobs")
                    print(f"   🗄️  Processed: {processed_count} jobs")
                    
                    if downloaded_count < len(completed_jobs):
                        print(f"   💡 {len(completed_jobs) - downloaded_count} jobs ready for download")
                    elif processed_count < downloaded_count:
                        print(f"   💡 {downloaded_count - processed_count} jobs ready for processing")
                    else:
                        print(f"   ✨ All completed jobs have been processed!")
                        
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

def cancel_uncompleted_jobs(config: Dict[str, Any]) -> None:
    """Cancel all uncompleted batch jobs to free up token quota."""
    print("🛑 Canceling uncompleted batch jobs...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline with proper context management
    try:
        with BatchEmbeddingPipeline(config) as pipeline:
            # Always discover jobs from OpenAI API (don't rely on checkpoints)
            print("📡 Discovering all batch jobs from OpenAI API...")
            
            try:
                # Discover all entity resolution batch jobs from OpenAI
                all_jobs = {}
                after = None
                total_discovered = 0
                
                # Paginate through ALL batch jobs to find ours
                while True:
                    if after:
                        batches = pipeline.openai_client.batches.list(limit=100, after=after)
                    else:
                        batches = pipeline.openai_client.batches.list(limit=100)
                    
                    total_discovered += len(batches.data)
                    
                    # Process this page of results
                    for batch in batches.data:
                        # Look for our batch jobs (filter by metadata)
                        metadata = getattr(batch, 'metadata', {})
                        if (metadata and 
                            metadata.get('created_by') == 'embedding_and_indexing_batch'):
                            all_jobs[batch.id] = batch
                    
                    # Check if there are more results
                    if not batches.has_more:
                        break
                        
                    # Get the last batch ID for pagination
                    if batches.data:
                        after = batches.data[-1].id
                    else:
                        break
                
                print(f"📊 Scanned {total_discovered} total batches, found {len(all_jobs)} entity resolution jobs")
                
                if not all_jobs:
                    print("ℹ️  No entity resolution batch jobs found in OpenAI API")
                    return
                
            except Exception as e:
                print(f"❌ Error discovering jobs from OpenAI API: {e}")
                return
            
            # Find jobs that can be cancelled
            print(f"📡 Checking status of {len(all_jobs)} discovered batch jobs...")
            
            cancellable_statuses = ['pending', 'validating', 'in_progress', 'finalizing']
            jobs_to_cancel = []
            completed_jobs = []
            already_failed = []
            cancelled_jobs = []
            
            for job_id, batch_status in all_jobs.items():
                try:
                    current_status = batch_status.status
                    
                    if current_status in cancellable_statuses:
                        jobs_to_cancel.append((job_id, current_status))
                    elif current_status == 'completed':
                        completed_jobs.append(job_id)
                    elif current_status == 'failed':
                        already_failed.append(job_id)
                    elif current_status == 'cancelled':
                        cancelled_jobs.append(job_id)
                        
                except Exception as e:
                    print(f"⚠️  Could not check job {job_id}: {e}")
        
        # Show summary
        print(f"\n📊 Job Status Summary:")
        print(f"   Can be cancelled: {len(jobs_to_cancel)}")
        print(f"   Already completed: {len(completed_jobs)}")
        print(f"   Already failed: {len(already_failed)}")
        print(f"   Already cancelled: {len(cancelled_jobs)}")
        
        if not jobs_to_cancel:
            print("ℹ️  No jobs found that can be cancelled")
            return
        
        # Confirm cancellation
        print(f"\n⚠️  About to cancel {len(jobs_to_cancel)} jobs:")
        for job_id, status in jobs_to_cancel[:5]:  # Show first 5
            print(f"   {job_id[:12]}... ({status})")
        if len(jobs_to_cancel) > 5:
            print(f"   ... and {len(jobs_to_cancel) - 5} more")
        
        print(f"\n🚨 WARNING: This will:")
        print(f"   • Cancel {len(jobs_to_cancel)} batch jobs")
        print(f"   • Free up token quota immediately")
        print(f"   • Jobs cannot be resumed once cancelled")
        
        # Get user confirmation
        try:
            response = input(f"\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Cancelled by user")
                return
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
        
        # Cancel jobs
        print(f"\n🛑 Cancelling {len(jobs_to_cancel)} jobs...")
        cancelled_count = 0
        failed_to_cancel = 0
        
        for job_id, status in jobs_to_cancel:
            try:
                # Cancel the job
                cancelled_batch = pipeline.openai_client.batches.cancel(job_id)
                
                # Update local tracking
                pipeline.batch_jobs[job_id]['status'] = 'cancelled'
                
                cancelled_count += 1
                print(f"   ✅ Cancelled {job_id[:12]}... ({status})")
                
            except Exception as e:
                failed_to_cancel += 1
                print(f"   ❌ Failed to cancel {job_id[:12]}...: {e}")
        
        # Update checkpoint if we have local tracking
        try:
            pipeline.load_checkpoint(checkpoint_dir)
            # Update status for cancelled jobs in local tracking
            for job_id, status in jobs_to_cancel[:cancelled_count]:
                if job_id in pipeline.batch_jobs:
                    pipeline.batch_jobs[job_id]['status'] = 'cancelled'
            pipeline.save_checkpoint(checkpoint_dir)
        except Exception as e:
            print(f"ℹ️  Could not update checkpoint (this is ok): {e}")
        
        print(f"\n📊 Cancellation Summary:")
        print(f"   Successfully cancelled: {cancelled_count}")
        print(f"   Failed to cancel: {failed_to_cancel}")
        print(f"   Token quota freed up immediately")
        
        if cancelled_count > 0:
            print(f"\n📋 Next steps:")
            print(f"   1. Check status: python batch_manager.py --status")
            print(f"   2. Resubmit failed jobs: python batch_manager.py --resubmit")
            print(f"   3. Or create new jobs: python batch_manager.py --create")
        
    except Exception as e:
        print(f"❌ Error cancelling jobs: {str(e)}")
        sys.exit(1)

def resubmit_failed_jobs(config: Dict[str, Any]) -> None:
    """Resubmit failed jobs that failed due to token limit."""
    print("🔄 Resubmitting failed batch jobs...")
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize batch pipeline
    pipeline = None
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Load existing jobs
        pipeline.load_checkpoint(checkpoint_dir)
        if not pipeline.batch_jobs:
            pipeline._recover_batch_jobs_from_api(checkpoint_dir)
        
        if not pipeline.batch_jobs:
            print("ℹ️  No batch jobs found")
            return
        
        # Check which jobs failed due to token limit
        print(f"📡 Checking status of batch jobs...")
        token_limit_failures = []
        
        for job_id, job_info in pipeline.batch_jobs.items():
            try:
                batch_status = pipeline.openai_client.batches.retrieve(job_id)
                if (batch_status.status == 'failed' and 
                    hasattr(batch_status, 'errors') and 
                    batch_status.errors):
                    
                    # Check if it failed due to token limit
                    for error in batch_status.errors.data:
                        if error.code == 'token_limit_exceeded':
                            token_limit_failures.append(job_id)
                            break
                            
            except Exception as e:
                print(f"⚠️  Could not check job {job_id}: {e}")
        
        if not token_limit_failures:
            print("ℹ️  No jobs failed due to token limit")
            return
        
        print(f"📊 Found {len(token_limit_failures)} jobs that failed due to token limit")
        
        # Check current quota usage
        active_jobs = 0
        for job_id, job_info in pipeline.batch_jobs.items():
            try:
                batch_status = pipeline.openai_client.batches.retrieve(job_id)
                if batch_status.status in ['pending', 'validating', 'in_progress', 'finalizing']:
                    active_jobs += 1
            except:
                pass
        
        print(f"📊 Currently {active_jobs} jobs active (consuming token quota)")
        
        if active_jobs > 10:  # Arbitrary threshold
            print("⏳ Too many active jobs. Wait for some to complete before resubmitting.")
            print("   Run this command again in 30-60 minutes.")
            return
        
        # Resubmit a small batch of failed jobs
        resubmit_count = min(5, len(token_limit_failures))  # Start with just 5
        jobs_to_resubmit = token_limit_failures[:resubmit_count]
        
        print(f"🚀 Attempting to resubmit {resubmit_count} failed jobs...")
        
        successful_resubmits = 0
        for job_id in jobs_to_resubmit:
            try:
                # Get the original batch to resubmit
                original_batch = pipeline.openai_client.batches.retrieve(job_id)
                
                # Create new batch with same input file
                new_batch = pipeline.openai_client.batches.create(
                    input_file_id=original_batch.input_file_id,
                    endpoint="/v1/embeddings",
                    completion_window="24h",
                    metadata={
                        "description": f"Resubmitted batch (original: {job_id})",
                        "created_by": "embedding_and_indexing_batch",
                        "resubmitted_from": job_id
                    }
                )
                
                # Track the new job
                pipeline.batch_jobs[new_batch.id] = {
                    'batch_idx': len(pipeline.batch_jobs),
                    'input_file_id': new_batch.input_file_id,
                    'status': 'submitted',
                    'created_at': new_batch.created_at,
                    'resubmitted_from': job_id,
                    'recovered': False
                }
                
                successful_resubmits += 1
                print(f"   ✅ Resubmitted {job_id} as {new_batch.id}")
                
            except Exception as e:
                if "token_limit_exceeded" in str(e):
                    print(f"   ⏳ Still at token limit. Stop here and try again later.")
                    break
                else:
                    print(f"   ❌ Failed to resubmit {job_id}: {e}")
        
        # Save updated jobs
        pipeline.save_checkpoint(checkpoint_dir)
        
        print(f"\n📊 Resubmission Summary:")
        print(f"   Successfully resubmitted: {successful_resubmits}")
        print(f"   Remaining failed jobs: {len(token_limit_failures) - successful_resubmits}")
        
        if successful_resubmits > 0:
            print(f"\n📋 Next steps:")
            print(f"   1. Wait 30-60 minutes for jobs to process")
            print(f"   2. Check status: python batch_manager.py --status")
            print(f"   3. Resubmit more: python batch_manager.py --resubmit")
        
    except Exception as e:
        print(f"❌ Error resubmitting jobs: {str(e)}")
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
            pipeline._recover_batch_jobs_from_api(checkpoint_dir)
        
        if not pipeline.batch_jobs:
            print("ℹ️  No batch jobs found to investigate")
            return
        
        # Check a sample of jobs to find failed ones (checking all 439 would take too long)
        sample_job_ids = list(pipeline.batch_jobs.keys())[:50]  # Check first 50 jobs
        print(f"📡 Checking current status of {len(sample_job_ids)} sample jobs with OpenAI...")
        failed_jobs = []
        
        for job_id in sample_job_ids:
            try:
                batch_status = pipeline.openai_client.batches.retrieve(job_id)
                if batch_status.status == 'failed':
                    failed_jobs.append((job_id, batch_status))
            except Exception as e:
                print(f"⚠️  Could not check status of job {job_id}: {e}")
        
        if not failed_jobs:
            print("ℹ️  No failed jobs found")
            return
        
        print(f"🔍 Investigating {len(failed_jobs)} failed jobs...")
        
        # Sample a few failed jobs to investigate
        sample_size = min(5, len(failed_jobs))
        sample_jobs = failed_jobs[:sample_size]
        
        for i, (job_id, batch_status) in enumerate(sample_jobs):
            print(f"\n--- Failed Job {i+1}/{sample_size}: {job_id} ---")
            
            try:
                
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
        pipeline._recover_batch_jobs_from_api(checkpoint_dir)
        
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
    _reset_embedding_stage_internal(config, preserve_tracking=False)

def reset_embedding_stage_selective(config: Dict[str, Any]) -> None:
    """Reset embedding stage but preserve processed hash tracking to avoid reprocessing."""
    print("🗑️  Resetting embedding_and_indexing stage (preserving tracking data)...")
    _reset_embedding_stage_internal(config, preserve_tracking=True)

def _save_processed_files_blacklist(config: Dict[str, Any], checkpoint_dir: str) -> int:
    """
    Save a blacklist of all batch job IDs that have been processed to avoid reprocessing them.
    This is used during reset to ignore previous attempts with flawed source data.
    """
    try:
        from src.embedding_and_indexing_batch import BatchEmbeddingPipeline
        pipeline = BatchEmbeddingPipeline(config)
        
        blacklisted_jobs = set()
        after = None
        total_jobs_scanned = 0
        
        # Scan ALL batch jobs (any status) to build blacklist
        while True:
            if after:
                batches = pipeline.openai_client.batches.list(limit=100, after=after)
            else:
                batches = pipeline.openai_client.batches.list(limit=100)
            
            total_jobs_scanned += len(batches.data)
            
            for batch in batches.data:
                # Check if this is an embedding batch job from our pipeline
                metadata = getattr(batch, 'metadata', {})
                endpoint = getattr(batch, 'endpoint', '')
                
                if (('/embeddings' in endpoint or endpoint == '/v1/embeddings') and
                    metadata and metadata.get('created_by') == 'embedding_and_indexing_batch'):
                    
                    # Add batch job ID to blacklist regardless of status
                    blacklisted_jobs.add(batch.id)
            
            if not batches.has_more:
                break
                
            if batches.data:
                after = batches.data[-1].id
            else:
                break
        
        pipeline.close()
        
        # Save blacklist to checkpoint
        if blacklisted_jobs:
            blacklist_path = os.path.join(checkpoint_dir, 'batch_blacklisted_files.pkl')
            import pickle
            with open(blacklist_path, 'wb') as f:
                pickle.dump(list(blacklisted_jobs), f)
        
        print(f"   📊 Scanned {total_jobs_scanned} total batch jobs from OpenAI")
        print(f"   🚫 Blacklisted {len(blacklisted_jobs)} batch job IDs to avoid reprocessing")
        
        return len(blacklisted_jobs)
        
    except Exception as e:
        print(f"   ❌ Error creating blacklist: {e}")
        return 0

def _reset_embedding_stage_internal(config: Dict[str, Any], preserve_tracking: bool = False) -> None:
    """Internal function to reset embedding stage with optional tracking preservation."""
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    files_deleted = []
    files_preserved = []
    weaviate_collection_deleted = False
    blacklisted_count = 0
    
    try:
        # FIRST: Create blacklist of all processed files to avoid reprocessing
        print("🚫 Creating blacklist of processed files to avoid reprocessing...")
        blacklisted_count = _save_processed_files_blacklist(config, checkpoint_dir)
        
        if blacklisted_count > 0:
            print(f"   ✅ Created blacklist with {blacklisted_count} files")
            print(f"   📋 Future processing will skip these files completely")
        else:
            print(f"   ℹ️  No files found to blacklist")
        
        # Second, try to delete the Weaviate collection
        print("🗄️  Dropping EntityString collection from Weaviate...")
        pipeline = None
        try:
            pipeline = BatchEmbeddingPipeline(config)
            
            # Check if collection exists and delete it
            try:
                pipeline.weaviate_client.collections.delete("EntityString")
                weaviate_collection_deleted = True
                print("   ✅ Deleted EntityString collection from Weaviate")
            except Exception as collection_error:
                if "not found" in str(collection_error).lower() or "does not exist" in str(collection_error).lower():
                    print("   ℹ️  EntityString collection does not exist in Weaviate")
                else:
                    print(f"   ⚠️  Could not delete EntityString collection: {collection_error}")
                    
        except Exception as weaviate_error:
            print(f"   ⚠️  Could not connect to Weaviate: {weaviate_error}")
        finally:
            if pipeline and hasattr(pipeline, 'weaviate_client'):
                try:
                    pipeline.weaviate_client.close()
                except:
                    pass
        
        # Define files to potentially delete
        real_time_files = [
            os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        ]
        
        batch_processing_files = [
            os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl'),
            os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        ]
        
        # Delete batch request/result files
        import glob
        batch_request_pattern = os.path.join(checkpoint_dir, 'batch_requests_*.jsonl')
        batch_result_pattern = os.path.join(checkpoint_dir, 'batch_results_*.jsonl')
        batch_files = glob.glob(batch_request_pattern) + glob.glob(batch_result_pattern)
        
        # Determine what to delete based on preserve_tracking flag
        if preserve_tracking:
            print("📁 Deleting job and file data (preserving hash tracking)...")
            # Preserve processed hashes but delete job tracking and files
            files_to_delete = [os.path.join(checkpoint_dir, 'batch_jobs.pkl')] + batch_files
            files_to_preserve = [
                os.path.join(checkpoint_dir, 'processed_hashes.pkl'),
                os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
            ]
        else:
            print("📁 Deleting all checkpoint files...")
            # Delete everything
            files_to_delete = real_time_files + batch_processing_files + batch_files
            files_to_preserve = []
        
        # Delete specified files
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                files_deleted.append(os.path.basename(file_path))
                print(f"   ✅ Deleted: {os.path.basename(file_path)}")
        
        # Track preserved files
        for file_path in files_to_preserve:
            if os.path.exists(file_path):
                files_preserved.append(os.path.basename(file_path))
                print(f"   💾 Preserved: {os.path.basename(file_path)}")
        
        # Summary
        print(f"\n🎉 Reset Summary:")
        if blacklisted_count > 0:
            print(f"   🚫 Blacklisted {blacklisted_count} processed files (will be ignored in future runs)")
        if weaviate_collection_deleted:
            print(f"   🗄️  Dropped EntityString collection from Weaviate")
        if files_deleted:
            print(f"   📁 Deleted {len(files_deleted)} files:")
            for filename in files_deleted:
                print(f"      • {filename}")
        if files_preserved:
            print(f"   💾 Preserved {len(files_preserved)} tracking files:")
            for filename in files_preserved:
                print(f"      • {filename}")
        if not files_deleted and not files_preserved:
            print("   ℹ️  No files found to delete or preserve")
        
        if preserve_tracking:
            print(f"\n📋 Selective reset complete. Previously processed strings will be skipped.")
            print(f"   Benefits:")
            print(f"   • Avoids reprocessing already embedded strings")
            print(f"   • Saves API costs and processing time")
            print(f"   • Maintains processing history")
        else:
            print(f"\n📋 Full reset complete. All tracking data cleared.")
        
        print(f"\n   You can now:")
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
  
  # Investigate failed jobs
  python batch_manager.py --investigate
  
  # Resubmit failed jobs (for token limit issues)
  python batch_manager.py --resubmit
  
  # Cancel all uncompleted jobs (frees token quota)
  python batch_manager.py --cancel
  
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
    action_group.add_argument('--reset-selective', action='store_true',
                             help='Reset embedding stage but preserve processed hash tracking to avoid reprocessing')
    action_group.add_argument('--recover', action='store_true',
                             help='Recover all batch jobs from OpenAI API (useful when checkpoints are corrupted)')
    action_group.add_argument('--investigate', action='store_true',
                             help='Investigate failed batch jobs to understand failure reasons')
    action_group.add_argument('--resubmit', action='store_true',
                             help='Resubmit failed batch jobs (useful for token limit failures)')
    action_group.add_argument('--cancel', action='store_true',
                             help='Cancel all uncompleted batch jobs (frees up token quota)')
    
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
    
    # For reset, recover, investigate, resubmit, and cancel commands, we don't need batch processing to be enabled
    if not args.reset and not args.reset_selective and not args.recover and not args.investigate and not args.resubmit and not args.cancel and not config.get('use_batch_embeddings', False):
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
        elif args.reset_selective:
            reset_embedding_stage_selective(config)
        elif args.recover:
            recover_all_jobs(config)
        elif args.investigate:
            investigate_failures(config)
        elif args.resubmit:
            resubmit_failed_jobs(config)
        elif args.cancel:
            cancel_uncompleted_jobs(config)
            
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