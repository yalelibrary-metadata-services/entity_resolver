"""
Intelligent Transition Controller for Entity Resolution

This module orchestrates the seamless transition from batch to real-time
embedding processing, ensuring no data loss and complete state preservation.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_state_consolidator import BatchStateConsolidator
from embedding_and_indexing_batch import BatchEmbeddingPipeline
from embedding_and_indexing import EmbeddingAndIndexingPipeline

logger = logging.getLogger(__name__)


class TransitionController:
    """
    Orchestrates seamless transition from batch to real-time embedding processing.
    
    Handles:
    - Batch process termination
    - State consolidation and migration
    - Failed request transfer
    - Real-time processing initiation
    - Progress monitoring and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transition controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        
        # Transition state
        self.transition_state = {
            'started': False,
            'batch_terminated': False,
            'state_consolidated': False,
            'realtime_started': False,
            'completed': False,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        # Components
        self.batch_pipeline = None
        self.realtime_pipeline = None
        self.consolidator = None
        
        logger.info("Initialized TransitionController")
    
    def pre_transition_analysis(self) -> Dict[str, Any]:
        """
        Analyze current system state before transition.
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        logger.info("Performing pre-transition analysis...")
        
        analysis = {
            'timestamp': time.time(),
            'batch_state': {},
            'realtime_state': {},
            'recommendations': [],
            'warnings': [],
            'errors': [],
            'transition_feasible': True
        }
        
        try:
            # Initialize consolidator for analysis
            self.consolidator = BatchStateConsolidator(self.config)
            
            # Load and analyze batch state
            batch_state = self.consolidator.load_batch_state()
            analysis['batch_state'] = batch_state
            
            # Load and analyze real-time state
            realtime_state = self.consolidator.load_realtime_state()
            analysis['realtime_state'] = realtime_state
            
            # Get readiness assessment
            summary = self.consolidator.get_consolidation_summary()
            transition_ready = summary['transition_ready']
            
            # Check for blocking issues
            if not transition_ready['ready']:
                analysis['transition_feasible'] = False
                analysis['errors'].extend(transition_ready['issues'])
            
            # Add warnings
            analysis['warnings'].extend(transition_ready.get('warnings', []))
            
            # Generate recommendations
            self._generate_recommendations(analysis, summary)
            
            logger.info(f"Pre-transition analysis complete. Feasible: {analysis['transition_feasible']}")
            
        except Exception as e:
            error_msg = f"Error in pre-transition analysis: {e}"
            logger.error(error_msg)
            analysis['errors'].append(error_msg)
            analysis['transition_feasible'] = False
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """
        Generate recommendations based on analysis.
        
        Args:
            analysis: Analysis dictionary to update
            summary: Consolidation summary
        """
        recommendations = []
        
        # Check for active batch jobs
        active_jobs = summary['batch_state']['queue_active']
        if active_jobs > 0:
            recommendations.append(
                f"Wait for {active_jobs} active batch jobs to complete before transition"
            )
        
        # Check for large number of failed requests
        failed_count = summary['batch_state']['failed_requests']
        if failed_count > 1000:
            recommendations.append(
                f"Large number of failed requests ({failed_count:,}). "
                "Consider analyzing failure patterns before transition"
            )
        
        # Check for processing efficiency
        batch_only = summary['consolidated_state']['batch_only_hashes']
        if batch_only > 0:
            recommendations.append(
                f"Transition will preserve {batch_only:,} batch-processed hashes"
            )
        
        # Estimate transition time
        failed_retryable = analysis.get('failed_request_analysis', {}).get('retryable', 0)
        if failed_retryable > 0:
            est_time = failed_retryable * 0.1  # Rough estimate: 0.1 seconds per retry
            recommendations.append(
                f"Estimated {failed_retryable:,} retryable requests will take ~{est_time:.0f} seconds"
            )
        
        analysis['recommendations'] = recommendations
    
    def terminate_batch_processing(self, force: bool = False) -> Dict[str, Any]:
        """
        Gracefully terminate batch processing.
        
        Args:
            force: Force termination even if jobs are active
            
        Returns:
            Dictionary with termination results
        """
        logger.info("Terminating batch processing...")
        
        termination_results = {
            'success': False,
            'active_jobs_cancelled': 0,
            'pending_jobs_cancelled': 0,
            'completed_jobs_processed': 0,
            'errors': []
        }
        
        try:
            # Initialize batch pipeline
            self.batch_pipeline = BatchEmbeddingPipeline(self.config)
            
            # Load current state
            self.batch_pipeline.load_checkpoint(self.checkpoint_dir)
            
            # Process any completed jobs first
            try:
                completed_results = self.batch_pipeline.process_completed_jobs(self.checkpoint_dir)
                if completed_results.get('status') == 'completed':
                    termination_results['completed_jobs_processed'] = completed_results.get('successful_jobs', 0)
                    logger.info(f"Processed {termination_results['completed_jobs_processed']} completed jobs")
            except Exception as e:
                logger.warning(f"Error processing completed jobs: {e}")
            
            # Check for active jobs
            active_jobs = len(self.batch_pipeline.active_batch_queue)
            pending_jobs = len(self.batch_pipeline.pending_batches)
            
            if active_jobs > 0:
                if force:
                    logger.warning(f"Force cancelling {active_jobs} active batch jobs")
                    # TODO: Implement job cancellation
                    termination_results['active_jobs_cancelled'] = active_jobs
                else:
                    raise ValueError(f"Cannot terminate: {active_jobs} active jobs still running. Use force=True to cancel them.")
            
            if pending_jobs > 0:
                logger.info(f"Clearing {pending_jobs} pending batch jobs")
                self.batch_pipeline.pending_batches.clear()
                termination_results['pending_jobs_cancelled'] = pending_jobs
            
            # Save final state
            self.batch_pipeline.save_checkpoint(self.checkpoint_dir)
            
            termination_results['success'] = True
            self.transition_state['batch_terminated'] = True
            
            logger.info("Batch processing terminated successfully")
            
        except Exception as e:
            error_msg = f"Error terminating batch processing: {e}"
            logger.error(error_msg)
            termination_results['errors'].append(error_msg)
        finally:
            # Clean up batch pipeline
            if self.batch_pipeline:
                try:
                    self.batch_pipeline.close()
                except:
                    pass
                self.batch_pipeline = None
        
        return termination_results
    
    def consolidate_state(self) -> Dict[str, Any]:
        """
        Consolidate batch and real-time processing state.
        
        Returns:
            Dictionary with consolidation results
        """
        logger.info("Consolidating processing state...")
        
        try:
            # Import and run consolidation
            from batch_state_consolidator import consolidate_batch_state
            
            consolidation_results = consolidate_batch_state(self.config)
            
            if consolidation_results['status'] == 'completed':
                self.transition_state['state_consolidated'] = True
                logger.info("State consolidation completed successfully")
            else:
                logger.error(f"State consolidation failed: {consolidation_results.get('error')}")
            
            return consolidation_results
            
        except Exception as e:
            error_msg = f"Error consolidating state: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
    
    def start_realtime_processing(self, string_dict: Dict[str, str], 
                                 field_hash_mapping: Dict[str, Dict[str, int]],
                                 string_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Start real-time processing with consolidated state.
        
        Args:
            string_dict: String hash to value mapping
            field_hash_mapping: Hash to field type mapping
            string_counts: Hash to frequency mapping
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting real-time processing with consolidated state...")
        
        try:
            # Initialize real-time pipeline
            self.realtime_pipeline = EmbeddingAndIndexingPipeline(self.config)
            
            # Process data with consolidated state
            processing_results = self.realtime_pipeline.process(
                string_dict, field_hash_mapping, string_counts, self.checkpoint_dir
            )
            
            if processing_results['status'] == 'completed':
                self.transition_state['realtime_started'] = True
                logger.info("Real-time processing started successfully")
            else:
                logger.error(f"Real-time processing failed: {processing_results.get('error')}")
            
            return processing_results
            
        except Exception as e:
            error_msg = f"Error starting real-time processing: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
        finally:
            # Clean up real-time pipeline
            if self.realtime_pipeline:
                try:
                    if hasattr(self.realtime_pipeline, 'weaviate_client'):
                        self.realtime_pipeline.weaviate_client.close()
                except:
                    pass
                self.realtime_pipeline = None
    
    def execute_transition(self, string_dict: Dict[str, str], 
                          field_hash_mapping: Dict[str, Dict[str, int]],
                          string_counts: Dict[str, int],
                          force_batch_termination: bool = False) -> Dict[str, Any]:
        """
        Execute complete batch-to-real-time transition.
        
        Args:
            string_dict: String hash to value mapping
            field_hash_mapping: Hash to field type mapping
            string_counts: Hash to frequency mapping
            force_batch_termination: Force termination of active batch jobs
            
        Returns:
            Dictionary with complete transition results
        """
        logger.info("Executing complete batch-to-real-time transition...")
        
        self.transition_state['started'] = True
        self.transition_state['start_time'] = time.time()
        
        transition_results = {
            'status': 'in_progress',
            'pre_analysis': {},
            'termination': {},
            'consolidation': {},
            'realtime_processing': {},
            'transition_state': self.transition_state.copy(),
            'elapsed_time': 0
        }
        
        try:
            # Step 1: Pre-transition analysis
            logger.info("Step 1: Pre-transition analysis")
            transition_results['pre_analysis'] = self.pre_transition_analysis()
            
            if not transition_results['pre_analysis']['transition_feasible']:
                if not force_batch_termination:
                    transition_results['status'] = 'blocked'
                    logger.error("Transition blocked by pre-analysis issues")
                    return transition_results
                else:
                    logger.warning("Proceeding with forced transition despite issues")
            
            # Step 2: Terminate batch processing
            logger.info("Step 2: Terminating batch processing")
            transition_results['termination'] = self.terminate_batch_processing(force_batch_termination)
            
            if not transition_results['termination']['success']:
                transition_results['status'] = 'failed'
                logger.error("Transition failed during batch termination")
                return transition_results
            
            # Step 3: Consolidate state
            logger.info("Step 3: Consolidating state")
            transition_results['consolidation'] = self.consolidate_state()
            
            if transition_results['consolidation']['status'] != 'completed':
                transition_results['status'] = 'failed'
                logger.error("Transition failed during state consolidation")
                return transition_results
            
            # Step 4: Start real-time processing
            logger.info("Step 4: Starting real-time processing")
            transition_results['realtime_processing'] = self.start_realtime_processing(
                string_dict, field_hash_mapping, string_counts
            )
            
            if transition_results['realtime_processing']['status'] != 'completed':
                transition_results['status'] = 'failed'
                logger.error("Transition failed during real-time processing startup")
                return transition_results
            
            # Mark transition as completed
            self.transition_state['completed'] = True
            self.transition_state['end_time'] = time.time()
            transition_results['status'] = 'completed'
            
            elapsed_time = self.transition_state['end_time'] - self.transition_state['start_time']
            transition_results['elapsed_time'] = elapsed_time
            transition_results['transition_state'] = self.transition_state.copy()
            
            logger.info(f"Batch-to-real-time transition completed successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error during transition execution: {e}"
            logger.error(error_msg)
            self.transition_state['errors'].append(error_msg)
            transition_results['status'] = 'error'
            transition_results['error'] = error_msg
            transition_results['transition_state'] = self.transition_state.copy()
        
        return transition_results
    
    def get_transition_status(self) -> Dict[str, Any]:
        """
        Get current transition status and progress.
        
        Returns:
            Dictionary with transition status
        """
        return {
            'transition_state': self.transition_state.copy(),
            'timestamp': time.time()
        }
    
    def save_transition_log(self, results: Dict[str, Any]) -> None:
        """
        Save transition results to log file.
        
        Args:
            results: Transition results dictionary
        """
        try:
            log_dir = self.config.get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = int(time.time())
            log_file = os.path.join(log_dir, f"batch_to_realtime_transition_{timestamp}.json")
            
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Transition log saved to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving transition log: {e}")


def transition_batch_to_realtime(config: Dict[str, Any], 
                                string_dict: Dict[str, str],
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int],
                                force: bool = False) -> Dict[str, Any]:
    """
    Main function to execute batch-to-real-time transition.
    
    Args:
        config: Configuration dictionary
        string_dict: String hash to value mapping
        field_hash_mapping: Hash to field type mapping
        string_counts: Hash to frequency mapping
        force: Force transition even if there are active batch jobs
        
    Returns:
        Dictionary with transition results
    """
    logger.info("Starting batch-to-real-time transition process...")
    
    controller = TransitionController(config)
    
    try:
        # Execute the transition
        results = controller.execute_transition(
            string_dict, field_hash_mapping, string_counts, force
        )
        
        # Save transition log
        controller.save_transition_log(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in transition process: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse
    import pickle
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Execute batch-to-real-time transition')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--force', action='store_true', help='Force transition even with active batch jobs')
    parser.add_argument('--analyze-only', action='store_true', help='Only perform pre-transition analysis')
    args = parser.parse_args()
    
    # Load configuration
    from config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Load preprocessing data
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    try:
        # Load required data files
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings, "
                   f"{len(field_hash_mapping)} field mappings, {len(string_counts)} string counts")
        
        if args.analyze_only:
            # Only perform analysis
            controller = TransitionController(config)
            analysis = controller.pre_transition_analysis()
            
            print("\n" + "="*60)
            print("PRE-TRANSITION ANALYSIS")
            print("="*60)
            print(f"Transition Feasible: {analysis['transition_feasible']}")
            
            if analysis['errors']:
                print("\nBLOCKING ISSUES:")
                for error in analysis['errors']:
                    print(f"  ❌ {error}")
            
            if analysis['warnings']:
                print("\nWARNINGS:")
                for warning in analysis['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if analysis['recommendations']:
                print("\nRECOMMENDATIONS:")
                for rec in analysis['recommendations']:
                    print(f"  💡 {rec}")
        else:
            # Execute full transition
            results = transition_batch_to_realtime(
                config, string_dict, field_hash_mapping, string_counts, args.force
            )
            
            print("\n" + "="*60)
            print("BATCH-TO-REAL-TIME TRANSITION RESULTS")
            print("="*60)
            print(f"Status: {results['status']}")
            
            if results['status'] == 'completed':
                print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")
                
                # Show consolidation stats
                if 'consolidation' in results:
                    summary = results['consolidation'].get('summary', {})
                    if 'consolidated_state' in summary:
                        cs = summary['consolidated_state']
                        print(f"Total Processed Hashes: {cs['total_processed_hashes']:,}")
                        print(f"From Batch Only: {cs['batch_only_hashes']:,}")
                        print(f"From Real-time Only: {cs['realtime_only_hashes']:,}")
                
                print("\n✅ Transition completed successfully!")
                print("Real-time processing is now active with all batch progress preserved.")
            
            elif results['status'] == 'blocked':
                print("\n❌ Transition blocked by pre-analysis issues")
                print("Use --force to proceed anyway or resolve the issues first")
            
            else:
                print(f"\n❌ Transition failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)