"""
Scaling Bridge Module

A streamlined interface between feature engineering and advanced scaling strategies.
Provides a coherent API for scaling selection, evaluation, and integration with 
minimal coupling and clear dependency boundaries.

Key Components:
1. Scaling strategy selection and evaluation
2. Feature engineering integration
3. Configuration management
4. Error recovery and fallback mechanisms

Architectural Design Principles:
- Single responsibility: Each method handles one specific scaling task
- Clear interface: Well-defined entry points for external components
- Robust error handling: Graceful degradation with appropriate fallbacks
- Performance optimization: Efficient vector processing with minimal overhead
"""

import logging
import numpy as np
import os
import types
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from src.scaling_integration import ScalingIntegration
from src.feature_engineering import FeatureEngineering
from src.robust_scaler import LibraryCatalogScaler

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Enumeration of supported scaling strategies for clear type safety."""
    LIBRARY_CATALOG = "library_catalog"
    ROBUST_MINMAX = "robust_minmax"  # Keeping for backward compatibility
    STANDARD_MINMAX = "standard_minmax"  # Deprecated
    STANDARD_SCALER = "standard_scaler"  # Deprecated


class ScalingBridge:
    """
    Bridge class providing a simplified interface between scaling functionality 
    and feature engineering pipeline.
    
    Implements a facade pattern to hide scaling complexity while providing a 
    consistent interface with robust error handling and performance optimization.
    """
    
    def __init__(self, config_path: str = "scaling_config.yml"):
        """
        Initialize the scaling bridge with configuration.
        
        Args:
            config_path: Path to scaling configuration file
        """
        self.config_path = config_path
        self.scaling = ScalingIntegration(config_path)
        self.feature_engineering = None
        self.feature_names = []
        self.scaler_state = {
            "is_active": False,
            "strategy": None,
            "feature_count": 0
        }
        logger.info(f"Initialized ScalingBridge with config from {config_path}")
    
    def connect(self, feature_engineering: FeatureEngineering) -> 'ScalingBridge':
        """
        Connect to feature engineering module with streamlined interface.
        
        Args:
            feature_engineering: Feature engineering instance to integrate with
            
        Returns:
            Self for method chaining
        """
        if feature_engineering is None:
            raise ValueError("Feature engineering instance cannot be None")
            
        self.feature_engineering = feature_engineering
        
        # Extract feature information
        if hasattr(feature_engineering, "feature_registry") and feature_engineering.feature_registry:
            self.feature_names = list(feature_engineering.feature_registry.keys())
        else:
            logger.warning("Feature registry not found or empty, using empty feature list")
            self.feature_names = []
        
        # Store original scaler for reference and potential restoration
        self.original_scaler = feature_engineering.scaler
        self.original_normalize_method = feature_engineering.normalize_features
        
        logger.info(f"Connected to feature engineering with {len(self.feature_names)} features")
        return self
    
    def evaluate(self, 
                feature_vectors: np.ndarray, 
                labels: Optional[np.ndarray] = None,
                auto_select: bool = True) -> Dict[str, Any]:
        """
        Evaluate scaling approaches and optionally select the best one.
        
        Args:
            feature_vectors: Feature vectors to evaluate scaling on
            labels: Optional ground truth labels for supervised evaluation
            auto_select: Whether to automatically select best approach
            
        Returns:
            Evaluation results dictionary with detailed metrics
        """
        # Validate inputs
        self._validate_feature_matrix(feature_vectors)
        
        # Ensure feature names match dimensions
        self._synchronize_feature_names(feature_vectors.shape[1])
        
        # Compare scaling approaches
        results = self.scaling.compare_scaling_approaches(feature_vectors, self.feature_names, labels)
        
        # Select best approach if requested
        if auto_select and results:
            strategy = self.scaling.select_best_approach(results)
            self.scaler_state["strategy"] = strategy
            logger.info(f"Selected optimal scaling strategy: {strategy}")
        
        # Generate visualization report in a standardized location
        self._generate_scaling_report()
        
        return results
    
    def apply(self, strategy: Optional[Union[str, ScalingStrategy]] = None) -> FeatureEngineering:
        """
        Apply selected scaling strategy to feature engineering.
        
        Args:
            strategy: Optional explicit strategy to use (overrides evaluated best strategy)
            
        Returns:
            Updated feature engineering instance
        """
        # Validate state
        if self.feature_engineering is None:
            raise ValueError("Feature engineering not connected. Call connect() first.")
        
        # Determine which strategy to use (explicit parameter > evaluated > default)
        if strategy is not None:
            # Handle enum values
            if isinstance(strategy, ScalingStrategy):
                strategy = strategy.value
            
            # Set and log the choice
            self.scaler_state["strategy"] = strategy
            logger.info(f"Using explicitly specified scaling strategy: {strategy}")
        elif not self.scaler_state.get("strategy"):
            # Default to library_catalog if no strategy set (standardized approach)
            self.scaler_state["strategy"] = "library_catalog"
            logger.info(f"Using default scaling strategy: library_catalog")
        
        # Get and validate scaler
        scaler = self._get_validated_scaler()
        if not scaler:
            logger.error(f"Could not obtain valid scaler for strategy: {self.scaler_state['strategy']}")
            return self.feature_engineering
            
        # Update feature engineering with new scaler
        self.feature_engineering.scaler = scaler
        self.feature_engineering.is_fitted = False
        
        # Replace normalization method with enhanced version
        self._inject_enhanced_normalization()
        
        # Update state
        self.scaler_state["is_active"] = True
        self.scaler_state["feature_count"] = len(self.feature_names)
        
        return self.feature_engineering
        
    def apply_with_fitted_scaler(self, strategy: Optional[Union[str, ScalingStrategy]] = None) -> FeatureEngineering:
        """
        Apply selected scaling strategy to feature engineering using pre-fitted scaler.
        Unlike apply(), this method doesn't re-fit the scaler, ensuring consistent scaling with training.
        
        Args:
            strategy: Optional explicit strategy to use (overrides evaluated best strategy)
            
        Returns:
            Updated feature engineering instance
        """
        # Validate state
        if self.feature_engineering is None:
            raise ValueError("Feature engineering not connected. Call connect() first.")
        
        # Set strategy if provided
        if strategy is not None:
            if isinstance(strategy, ScalingStrategy):
                strategy = strategy.value
            self.scaler_state["strategy"] = strategy
            logger.info(f"Using explicitly specified scaling strategy: {strategy}")
        elif not self.scaler_state.get("strategy"):
            self.scaler_state["strategy"] = "library_catalog"
            logger.info(f"Using default scaling strategy: library_catalog")
        
        # Validate scaler is already fitted
        if not hasattr(self.feature_engineering, 'scaler') or not hasattr(self.feature_engineering.scaler, 'is_fitted'):
            logger.warning("Scaler is not fitted; consistency with training cannot be guaranteed")
        elif not getattr(self.feature_engineering.scaler, 'is_fitted', False):
            logger.warning("Scaler reports it is not fitted; consistency with training cannot be guaranteed")
            
        # Inject enhanced normalization method that ensures fitted scaler is used
        self._inject_enhanced_normalization_without_refitting()
        
        # Update state
        self.scaler_state["is_active"] = True
        self.scaler_state["feature_count"] = len(self.feature_names)
        
        return self.feature_engineering
    
    def reset(self) -> FeatureEngineering:
        """
        Reset feature engineering to original scaling state.
        
        Returns:
            Feature engineering instance with original scaling
        """
        if self.feature_engineering is None:
            logger.warning("No feature engineering instance to reset")
            return None
            
        if not self.scaler_state.get("is_active", False):
            logger.info("No active scaling to reset")
            return self.feature_engineering
            
        # Restore original components
        self.feature_engineering.scaler = self.original_scaler
        self.feature_engineering.is_fitted = False
        self.feature_engineering.normalize_features = types.MethodType(
            self.original_normalize_method.__func__, 
            self.feature_engineering
        )
        
        # Update state
        self.scaler_state["is_active"] = False
        
        logger.info("Successfully reset to original scaling configuration")
        return self.feature_engineering
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get scaling configuration metadata.
        
        Returns:
            Dictionary with detailed scaling metadata
        """
        # Basic metadata
        metadata = {
            'strategy': self.scaler_state.get("strategy"),
            'is_active': self.scaler_state.get("is_active", False),
            'feature_count': len(self.feature_names),
            'config_path': self.config_path
        }
        
        # Add scaler-specific information if active
        if self.scaler_state.get("is_active") and self.feature_engineering and hasattr(self.feature_engineering, "scaler"):
            scaler = self.feature_engineering.scaler
            
            # Get scaler parameters based on type
            if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                metadata["scaling_range"] = {
                    "min": scaler.data_min_.tolist() if hasattr(scaler.data_min_, "tolist") else scaler.data_min_,
                    "max": scaler.data_max_.tolist() if hasattr(scaler.data_max_, "tolist") else scaler.data_max_
                }
            
            # Add configuration parameters
            if hasattr(scaler, "clip_percentile"):
                metadata["clip_percentile"] = scaler.clip_percentile
                
            if hasattr(scaler, "feature_groups"):
                metadata["has_feature_groups"] = True
                
        return metadata
    
    def _validate_feature_matrix(self, feature_vectors: np.ndarray) -> None:
        """
        Validate feature matrix for scaling operations.
        
        Args:
            feature_vectors: Feature vectors to validate
            
        Raises:
            ValueError: If feature matrix is invalid
        """
        if not isinstance(feature_vectors, np.ndarray):
            raise TypeError(f"Feature vectors must be numpy array, got {type(feature_vectors)}")
            
        if feature_vectors.ndim != 2:
            raise ValueError(f"Feature vectors must be 2D, got {feature_vectors.ndim}D")
            
        if feature_vectors.shape[0] == 0:
            raise ValueError("Feature vector array is empty (no samples)")
            
        if feature_vectors.shape[1] == 0:
            raise ValueError("Feature vector array has no features (zero width)")
    
    def _synchronize_feature_names(self, feature_count: int) -> None:
        """
        Ensure feature names match the expected feature count.
        
        Args:
            feature_count: Expected number of features
        """
        if len(self.feature_names) != feature_count:
            logger.warning(f"Feature names count ({len(self.feature_names)}) doesn't match "
                          f"expected feature count ({feature_count})")
            
            # Adjust feature names list length
            if len(self.feature_names) > feature_count:
                # Truncate if too many names
                self.feature_names = self.feature_names[:feature_count]
                logger.info(f"Truncated feature names list to {feature_count} entries")
            else:
                # Extend with generic names if too few
                additional = [f"feature_{i}" for i in range(len(self.feature_names), feature_count)]
                self.feature_names.extend(additional)
                logger.info(f"Extended feature names with {len(additional)} generic entries")
    
    def _get_validated_scaler(self) -> Any:
        """
        Get and validate scaler for selected strategy.
        
        Returns:
            Valid scaler instance or None if validation fails
        """
        strategy = self.scaler_state.get("strategy")
        if not strategy:
            logger.error("No scaling strategy selected")
            return None
            
        # Get scaler from integration module
        scaler = self.scaling.get_scaler(strategy)
        if scaler is None:
            logger.error(f"Failed to obtain scaler for strategy: {strategy}")
            return None
            
        # Basic validation
        if not hasattr(scaler, "fit") or not hasattr(scaler, "transform"):
            logger.error(f"Invalid scaler: missing required fit/transform methods")
            return None
            
        return scaler
    
    def _inject_enhanced_normalization(self) -> None:
        """
        Inject enhanced normalization method into feature engineering.
        Uses a clean implementation with proper error handling and fallbacks.
        """
        # Store key references in local variables to avoid closure issues
        strategy = self.scaler_state.get("strategy")
        feature_names = self.feature_names.copy()  # Avoid reference issues
        original_normalize = self.original_normalize_method
        
        # Define enhanced normalization function
        def enhanced_normalize(self, feature_vectors, fit=False):
            """
            Enhanced feature normalization with improved scaling and error handling.
            
            Args:
                feature_vectors: Feature vectors to normalize
                fit: Whether to fit the scaler on this data
                
            Returns:
                Normalized feature vectors
            """
            # Early exit for empty input
            if feature_vectors.size == 0:
                return feature_vectors
                
            # Feature vector validation and preprocessing
            if not isinstance(feature_vectors, np.ndarray):
                try:
                    feature_vectors = np.array(feature_vectors, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to convert input to numpy array: {e}")
                    return feature_vectors
            
            # Handle NaN and infinity values
            if np.isnan(feature_vectors).any() or np.isinf(feature_vectors).any():
                logger.warning("Input contains NaN/Inf values, applying automatic cleaning")
                feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Scaling with error handling
            try:
                # Use appropriate method based on strategy
                if fit or not self.is_fitted:
                    # Fit transform based on strategy type
                    # Call the appropriate fit_transform method for the scaler
                    if hasattr(self.scaler, 'fit_transform') and callable(getattr(self.scaler, 'fit_transform')):
                        if strategy == 'library_catalog' or isinstance(self.scaler, LibraryCatalogScaler):
                            X_scaled = self.scaler.fit_transform(feature_vectors, feature_names)
                        else:
                            X_scaled = self.scaler.fit_transform(feature_vectors)
                    
                    self.is_fitted = True
                    logger.debug(f"Fitted {strategy} scaler with input data")
                else:
                    # Transform only
                    # Call the appropriate transform method for the scaler
                    if hasattr(self.scaler, 'transform') and callable(getattr(self.scaler, 'transform')):
                        if strategy == 'library_catalog' or isinstance(self.scaler, LibraryCatalogScaler):
                            X_scaled = self.scaler.transform(feature_vectors)
                        else:
                            X_scaled = self.scaler.transform(feature_vectors)
                
                # Validate output - ensure no NaN/Inf values
                if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                    logger.warning("Scaling produced invalid values, applying correction")
                    X_scaled = np.nan_to_num(X_scaled, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Ensure output is in expected range [0,1]
                X_scaled = np.clip(X_scaled, 0.0, 1.0)
                
                # CRITICAL FIX: Validate and fix binary features, ensuring consistent handling
                # This is especially important for person_low_levenshtein_indicator and person_low_jaro_winkler_indicator
                # which are crucial for matching entities like '16044224#Agent700-36' and '7732682#Agent700-29'
                if hasattr(self.scaler, '_validate_binary_features'):
                    # If scaler has built-in binary validation
                    X_scaled = self.scaler._validate_binary_features(X_scaled, feature_vectors)
                elif hasattr(self, '_owner') and hasattr(self._owner, '_validate_binary_features'):
                    # If we're in a method bound to feature_engineering with parent ScalingBridge
                    X_scaled = self._owner._validate_binary_features(X_scaled, feature_vectors, feature_names)
                else:
                    # Fallback direct validation for critical features if custom methods not available
                    # Find and validate key binary indicator features
                    if feature_names is not None:
                        for i, name in enumerate(feature_names):
                            if i < X_scaled.shape[1] and ("indicator" in name.lower() or "match" in name.lower()):
                                # Force exact binary values (0.0 or 1.0)
                                orig_col = feature_vectors[:, i]
                                binary_values = np.zeros_like(orig_col)
                                binary_values[orig_col >= 0.5] = 1.0
                                X_scaled[:, i] = binary_values
                
                return X_scaled
                
            except Exception as e:
                logger.error(f"Error in enhanced scaling: {str(e)}, using fallback")
                
                # Fall back to original normalization method
                try:
                    return original_normalize(self, feature_vectors, fit=fit)
                except Exception as fallback_error:
                    logger.error(f"Fallback scaling also failed: {fallback_error}")
                    # Last resort: return input with simple clipping
                    return np.clip(feature_vectors, 0.0, 1.0)
        
        # Bind enhanced method to feature engineering instance
        self.feature_engineering.normalize_features = types.MethodType(
            enhanced_normalize, self.feature_engineering
        )
        
        # CRITICAL FIX: Set owner reference in feature engineering to access validation methods
        self.feature_engineering._owner = self
        
        logger.info(f"Injected enhanced normalization method with {strategy} strategy")
    
    def _inject_enhanced_normalization_without_refitting(self) -> None:
        """
        Inject enhanced normalization method that never refits the scaler.
        This ensures consistent scaling with training.
        """
        # Store key references for closure
        strategy = self.scaler_state.get("strategy")
        feature_names = self.feature_names.copy()
        original_normalize = self.original_normalize_method
        
        # Store binary features list for validation
        binary_features = []
        for i, name in enumerate(feature_names):
            if "indicator" in name.lower() or "match" in name.lower():
                binary_features.append(name)
        
        def enhanced_normalize_production(self, feature_vectors, fit=False):
            """
            Enhanced normalization for production that never refits.
            Always uses the pre-fitted scaler from training.
            
            Args:
                feature_vectors: Feature vectors to normalize
                fit: Ignored - always uses pre-fitted scaler
                
            Returns:
                Normalized feature vectors
            """
            # Validation and preprocessing
            if feature_vectors.size == 0:
                return feature_vectors
                
            if not isinstance(feature_vectors, np.ndarray):
                try:
                    feature_vectors = np.array(feature_vectors, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Failed to convert input to numpy array: {e}")
                    return feature_vectors
            
            # Handle NaN and infinity values
            if np.isnan(feature_vectors).any() or np.isinf(feature_vectors).any():
                logger.warning("Input contains NaN/Inf values, applying automatic cleaning")
                feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=1.0, neginf=0.0)
            
            # CRITICAL: Never fit, only transform
            try:
                # Always use transform, never fit
                if hasattr(self.scaler, 'transform') and callable(getattr(self.scaler, 'transform')):
                    # Force transform without fitting
                    if strategy == 'library_catalog' or isinstance(self.scaler, LibraryCatalogScaler):
                        X_scaled = self.scaler.transform(feature_vectors)
                    else:
                        X_scaled = self.scaler.transform(feature_vectors)
                    
                    logger.debug(f"Applied pre-fitted {strategy} scaler to feature vectors (transform only)")
                else:
                    logger.error(f"Scaler missing transform method")
                    # Fallback to original normalization
                    return original_normalize(self, feature_vectors, fit=False)
                
                # Check for any NaN or inf values in result
                if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                    logger.warning("Scaling produced invalid values, applying correction")
                    X_scaled = np.nan_to_num(X_scaled, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Ensure output is in expected range [0,1]
                X_scaled = np.clip(X_scaled, 0.0, 1.0)
                
                # Ensure binary features preservation
                if feature_names:
                    for i, name in enumerate(feature_names):
                        if i < feature_vectors.shape[1] and name in binary_features:
                            # Get original values
                            orig_values = feature_vectors[:, i]
                            # Create exact binary values
                            binary_values = np.zeros_like(orig_values)
                            binary_values[orig_values >= 0.5] = 1.0
                            
                            # Update the normalized array with exact binary values
                            X_scaled[:, i] = binary_values
                            
                            # Special diagnostic for test case
                            test_entity_pair = getattr(self, '_current_entity_pair', None)
                            if test_entity_pair and ("16044224#Agent700-36" in test_entity_pair and "7732682#Agent700-29" in test_entity_pair):
                                logger.info(f"Critical test pair binary feature '{name}' preserved exactly: {binary_values[0]}")
                
                return X_scaled
                
            except Exception as e:
                logger.error(f"Error in production scaling: {str(e)}")
                
                # Fall back to original normalization without fitting
                try:
                    return original_normalize(self, feature_vectors, fit=False)
                except Exception as fallback_error:
                    logger.error(f"Fallback scaling also failed: {fallback_error}")
                    # Last resort: return input with simple clipping
                    return np.clip(feature_vectors, 0.0, 1.0)
        
        # Bind enhanced method to feature engineering instance
        self.feature_engineering.normalize_features = types.MethodType(
            enhanced_normalize_production, self.feature_engineering
        )
        
        # Set owner reference in feature engineering
        self.feature_engineering._owner = self
        
        logger.info(f"Injected production normalization method with pre-fitted {strategy} scaler (no refitting)")
    
    def _validate_binary_features(self, X_scaled, X_original, feature_names=None):
        """
        Critical validation of binary features to ensure they are preserved during scaling.
        This method ensures binary indicator features are consistently handled, which is
        crucial for reliable entity matching.
        
        Args:
            X_scaled: Scaled feature vectors
            X_original: Original feature vectors
            feature_names: Optional feature names for detailed logging
        """
        if X_scaled.shape != X_original.shape:
            logger.error(f"Cannot validate binary features: shape mismatch {X_scaled.shape} vs {X_original.shape}")
            return
        
        # Default feature names if not provided
        if feature_names is None:
            if hasattr(self.feature_engineering, "feature_registry") and self.feature_engineering.feature_registry:
                feature_names = list(self.feature_engineering.feature_registry.keys())
            else:
                feature_names = [f"feature_{i}" for i in range(X_scaled.shape[1])]
                
        # Find binary indicator features
        binary_features = []
        binary_indices = []
        
        for i, name in enumerate(feature_names):
            if i < X_scaled.shape[1] and ("indicator" in name.lower() or "match" in name.lower()):
                binary_features.append(name)
                binary_indices.append(i)
                
        if not binary_indices:
            return  # No binary features to validate
            
        # Check and fix each binary feature
        fixed_count = 0
        for idx, name in zip(binary_indices, binary_features):
            orig_col = X_original[:, idx]
            scaled_col = X_scaled[:, idx]
            
            # Find values that should be binary but aren't exactly 0.0 or 1.0
            # We check if the scaled values differ significantly from binary values
            should_be_zero = (orig_col < 0.5) & ((scaled_col != 0.0) & (scaled_col > 0.001))
            should_be_one = (orig_col >= 0.5) & ((scaled_col != 1.0) & (scaled_col < 0.999))
            
            problem_count = np.sum(should_be_zero) + np.sum(should_be_one)
            if problem_count > 0:
                logger.warning(f"BINARY VALIDATION: Fixed {problem_count} values in '{name}' that were not exactly binary")
                
                # Force binary values
                X_scaled[should_be_zero, idx] = 0.0
                X_scaled[should_be_one, idx] = 1.0
                fixed_count += problem_count
                
        if fixed_count > 0:
            logger.info(f"BINARY VALIDATION: Fixed a total of {fixed_count} binary indicator values across {len(binary_features)} features")
        
        return X_scaled
    
    def _generate_scaling_report(self) -> str:
        """
        Generate scaling visualization report in standardized location.
        
        Returns:
            Path to generated report or None if generation failed
        """
        try:
            # Create output directory
            output_dir = os.path.join('data', 'output', 'scaling_viz')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report
            report_path = self.scaling.save_scaling_report()
            
            if report_path and os.path.exists(report_path):
                logger.info(f"Generated scaling report at {report_path}")
                return report_path
            else:
                logger.warning("Failed to generate scaling report")
                return None
                
        except Exception as e:
            logger.error(f"Error generating scaling report: {e}")
            return None
