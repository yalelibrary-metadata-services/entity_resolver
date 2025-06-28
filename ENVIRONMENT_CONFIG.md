# Environment-Specific Configuration

The Entity Resolution Pipeline supports environment-specific configurations to optimize resource usage for different deployment scenarios.

## Overview

The pipeline automatically detects the runtime environment and applies appropriate resource allocations:

- **Local Development**: Moderate resource usage suitable for development and testing
- **Production**: High-performance settings optimized for 64-core, 247GB RAM servers

## Usage

### Production Mode
```bash
# Set environment variable
export PIPELINE_ENV=prod

# Run with production settings
./run_prod.sh

# Or directly
python -m src.orchestrating
```

### Local Development Mode (Default)
```bash
# Set environment variable (optional, local is default)
export PIPELINE_ENV=local

# Run with local settings
./run_local.sh

# Or directly
python -m src.orchestrating
```

## Configuration Differences

| Setting | Local | Production | Purpose |
|---------|-------|------------|---------|
| `preprocessing_workers` | 4 | 32 | Parallel file processing |
| `feature_workers` | 4 | 48 | Feature computation parallelism |
| `classification_workers` | 8 | 32 | Entity classification parallelism |
| `weaviate_batch_size` | 100 | 1000 | Vector database batch size |
| `string_cache_size` | 100,000 | 2,000,000 | String lookup cache |
| `vector_cache_size` | 50,000 | 1,000,000 | Vector cache size |
| `similarity_cache_size` | 200,000 | 5,000,000 | Similarity computation cache |

## Architecture

### Shared Configuration Utility
All pipeline modules use `src/config_utils.py` for consistent environment detection:

```python
from src.config_utils import load_config_with_environment

# Automatically applies environment-specific overrides
config = load_config_with_environment('config.yml')
```

### Environment Detection
The system checks the `PIPELINE_ENV` environment variable:
- `PIPELINE_ENV=prod` → Production settings
- `PIPELINE_ENV=local` → Local settings (default)
- Any other value → Defaults to local with warning

## Modules Supporting Environment Configuration

### Core Pipeline Modules
- ✅ `src/orchestrating.py` - Pipeline orchestration
- ✅ `src/preprocessing.py` - Data preprocessing
- ✅ `src/embedding_and_indexing.py` - Vector generation and indexing
- ✅ `src/embedding_and_indexing_batch.py` - Batch vector processing
- ✅ `src/feature_engineering.py` - Feature computation
- ✅ `src/training.py` - Model training
- ✅ `src/classifying.py` - Entity classification
- ✅ `src/querying.py` - Vector querying
- ✅ `src/indexing.py` - Legacy indexing

### Subject Enhancement Modules
- ✅ `src/subject_quality.py` - Subject quality audit
- ✅ `src/subject_imputation.py` - Subject imputation

### Configuration Propagation
- **Orchestrator**: Applies environment config and passes to all modules
- **Standalone Modules**: Use shared config utility for consistency
- **Weaviate Client**: Receives environment-specific connection settings

## Testing

Run the environment configuration test:
```bash
python test_environment_config.py
```

This verifies:
- Environment detection works correctly
- Local vs production settings differ appropriately
- Invalid environments default to local
- All configuration keys are properly overridden

## Performance Impact

### Local Development
- Moderate resource usage (~4-8 workers)
- Suitable for laptops and development machines
- Smaller cache sizes to conserve memory

### Production (64 cores, 247GB RAM)
- High parallelism (32-48 workers)
- Large batch sizes for efficiency
- Massive caches for performance
- Optimized for large-scale processing

## Adding New Environment-Specific Settings

To add new environment-specific configuration:

1. **Add to config.yml**:
```yaml
# Local settings
local_resources:
  new_setting: local_value

# Production settings  
prod_resources:
  new_setting: prod_value

# Active setting (default)
new_setting: local_value
```

2. **Update config_utils.py** if needed:
```python
# Add to apply_environment_config() if special handling required
config.update({
    'new_setting': resource_config.get('new_setting')
})
```

3. **Use in modules**:
```python
new_value = config.get('new_setting', default_value)
```

## Troubleshooting

### Environment Not Applied
- Check `PIPELINE_ENV` environment variable is set correctly
- Verify module uses `load_config_with_environment()` not direct YAML loading
- Check logs for environment detection messages

### Performance Issues
- Local mode: Increase worker counts in `local_resources` section
- Production mode: Verify system has sufficient cores/RAM for production settings
- Monitor system resources during pipeline execution

### Cache Memory Usage
- Production caches can use several GB of RAM
- Adjust cache sizes in prod_cache section if memory constraints exist