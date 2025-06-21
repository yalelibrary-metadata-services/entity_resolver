"""
Checkpoint Management Module for Entity Resolution

This module provides a centralized checkpoint tracking system for the entity resolution
pipeline, enabling deterministic resumption from the last successful stage.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class PipelineStateTracker:
    """
    Tracks and manages pipeline execution state for checkpoint resumption.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize the pipeline state tracker.
        
        Args:
            checkpoint_dir: Directory for saving checkpoint state
        """
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, 'pipeline_state.json')
        self.state = self._load_state()
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.debug(f"Initialized PipelineStateTracker with state file: {self.state_file}")
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load pipeline state from state file.
        
        Returns:
            Dictionary with pipeline state
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.debug(f"Loaded pipeline state: {state}")
                return state
            except Exception as e:
                logger.error(f"Error loading pipeline state: {str(e)}")
        
        # Initialize default state
        return {
            'last_completed_stage': None,
            'stage_timestamps': {},
            'stage_metrics': {},
            'version': '1.0'
        }
    
    def update_stage_completion(self, stage_name: str, status: str = 'completed', 
                               metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the pipeline state when a stage completes.
        
        Args:
            stage_name: Name of the completed stage
            status: Status of the stage ('completed', 'failed', 'skipped')
            metrics: Optional stage metrics
        """
        self.state['last_completed_stage'] = stage_name
        self.state['stage_timestamps'][stage_name] = time.time()
        
        if metrics:
            self.state['stage_metrics'][stage_name] = metrics
        
        self._save_state()
        logger.debug(f"Updated pipeline state for stage: {stage_name}, status: {status}")
    
    def _save_state(self) -> None:
        """
        Save pipeline state to state file.
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved pipeline state to: {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving pipeline state: {str(e)}")
    
    def get_last_completed_stage(self) -> Optional[str]:
        """
        Get the name of the last successfully completed stage.
        
        Returns:
            Name of the last completed stage or None if no stage has completed
        """
        return self.state.get('last_completed_stage')
    
    def get_next_stage(self, stages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get the name of the next stage to run based on the last completed stage.
        
        Args:
            stages: List of stage dictionaries with 'name' keys
            
        Returns:
            Name of the next stage to run or None if all stages are completed
        """
        last_stage = self.get_last_completed_stage()
        
        if not last_stage:
            # No stages completed, start from the first enabled stage
            for stage in stages:
                if stage.get('enabled', True):
                    return stage['name']
            return None
        
        # Find the index of the last completed stage
        stage_names = [stage['name'] for stage in stages if stage.get('enabled', True)]
        try:
            last_index = stage_names.index(last_stage)
            if last_index + 1 < len(stage_names):
                return stage_names[last_index + 1]
        except ValueError:
            logger.warning(f"Could not find last completed stage: {last_stage}")
        
        return None
    
    def reset_stage(self, stage_name: str) -> None:
        """
        Reset the state for a specific stage.
        
        Args:
            stage_name: Name of the stage to reset
        """
        if stage_name in self.state['stage_timestamps']:
            del self.state['stage_timestamps'][stage_name]
        
        if stage_name in self.state['stage_metrics']:
            del self.state['stage_metrics'][stage_name]
        
        # If this was the last completed stage, update to the previous one
        if self.state['last_completed_stage'] == stage_name:
            # Find the previous stage that has a timestamp
            previous_stage = None
            stage_times = sorted(
                [(stage, timestamp) for stage, timestamp in self.state['stage_timestamps'].items()],
                key=lambda x: x[1]
            )
            
            if stage_times:
                previous_stage = stage_times[-1][0]
            
            self.state['last_completed_stage'] = previous_stage
        
        self._save_state()
        logger.debug(f"Reset state for stage: {stage_name}")
    
    def reset_all(self) -> None:
        """
        Reset the entire pipeline state.
        """
        self.state = {
            'last_completed_stage': None,
            'stage_timestamps': {},
            'stage_metrics': {},
            'version': '1.0'
        }
        self._save_state()
        logger.debug("Reset all pipeline state")
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """
        Get the overall progress of the pipeline.
        
        Returns:
            Dictionary with pipeline progress information
        """
        return {
            'last_completed_stage': self.state.get('last_completed_stage'),
            'completed_stages': list(self.state.get('stage_timestamps', {}).keys()),
            'stage_metrics': self.state.get('stage_metrics', {})
        }

def get_checkpoint_manager(config: Dict[str, Any]) -> PipelineStateTracker:
    """
    Factory function to create or get a checkpoint manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized PipelineStateTracker
    """
    checkpoint_dir = config.get('checkpoint_dir', 'data/checkpoints')
    return PipelineStateTracker(checkpoint_dir)
