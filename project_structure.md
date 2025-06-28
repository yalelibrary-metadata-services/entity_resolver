# Entity Resolution Pipeline: Complete Project Structure

## Overview

This entity resolution pipeline processes Yale University Library Catalog data (derived from MARC 21 records) to identify and resolve entity matches across catalog entries. The system combines vector embeddings, feature engineering, and machine learning to achieve **99.55% precision** and **82.48% recall** in production.

## Pipeline Architecture

The system implements a **triple-layer approach**:

1. **Main Pipeline**: Entity resolution using logistic regression with engineered features
2. **Auxiliary Classification**: Individual record taxonomy classification using parallel API processing and SetFit models
3. **Subject Enhancement**: Automated quality audit and imputation for subject fields using vector similarity

### Core Pipeline Stages

```
Input CSV → Preprocessing → Embedding & Indexing → Subject Enhancement → Training → Classification → Reporting
     ↓            ↓              ↓                  ↓                ↓            ↓            ↓
  Hash-based   OpenAI API    Feature Eng.      Quality Audit    Similarity   Clustering   HTML/CSV
  Dedup        Weaviate DB   Scaling           Imputation       Matching     Results      Reports
                Batch API     Environment-      Vector Join      Enhanced     Enhanced     Subject
                (50% cost)    Specific Config   Centroid Calc    Features     Clusters     Analysis
```

## Project Directory Structure

```
entity_resolver/
├── 📁 Core Pipeline Components
│   ├── main.py                          # CLI entry point with stage control
│   ├── batch_manager.py                 # Intelligent batch processing management (automated/manual modes)
│   ├── config.yml                       # Main pipeline configuration
│   ├── scaling_config.yml               # Feature scaling configuration
│   ├── docker-compose.yml               # Weaviate vector database setup
│   └── requirements.txt                 # Python dependencies
│
├── 📁 src/                              # Core pipeline modules
│   ├── orchestrating.py                # Pipeline orchestration & stage management
│   ├── preprocessing.py                # Optimized CSV processing & CRC32-based deduplication
│   ├── embedding_and_indexing.py       # Real-time OpenAI embeddings & Weaviate integration
│   ├── embedding_and_indexing_batch.py # Fully Automated Batch OpenAI API (50% cost savings, self-managing 16-batch queue, intelligent quota management)
│   ├── subject_quality.py              # Subject quality audit using vector similarity analysis
│   ├── subject_imputation.py           # Missing subject imputation via composite field vectors
│   ├── feature_engineering.py          # Feature calculation & caching system
│   ├── training.py                     # Logistic regression classifier training
│   ├── classifying.py                  # Entity matching & transitive clustering
│   ├── reporting.py                    # Metrics, visualizations & HTML dashboards
│   ├── scaling_bridge.py               # Feature scaling coordination
│   ├── robust_scaler.py                # Domain-specific scaling strategies
│   ├── checkpoint_manager.py           # Pipeline state persistence & resumption
│   ├── taxonomy_feature.py             # SetFit taxonomy integration
│   ├── birth_death_regexes.py          # Temporal entity matching patterns
│   ├── custom_features.py              # Extensible feature registration system
│   ├── config_utils.py                 # Environment-specific configuration management
│   ├── utils.py                        # Utilities & helper functions
│   ├── visualization.py                # Feature importance & performance plots
│   └── vector_diagnostics.py           # Vector similarity debugging tools
│
├── 📁 scripts/                         # Individual Record Classification System
│   ├── verify_individual_classifications.py         # Sequential record classification
│   └── verify_individual_classifications_parallel.py # Parallel API processing with rate limiting
│
├── 📁 setfit/                          # SetFit Hierarchical Taxonomy Classification
│   ├── train_setfit_classifier.py      # SetFit model training pipeline
│   ├── predict_setfit_classifier.py    # SetFit prediction & classification
│   ├── train_setfit_simple.py          # Simplified SetFit training (recently updated)
│   ├── train_setfit_*.py               # Additional training variants (memory, server, etc.)
│   ├── SETFIT_README.md               # SetFit system documentation
│   ├── setfit_requirements.txt         # SetFit-specific dependencies
│   └── model_multilingual_minilm/      # Trained SetFit model artifacts
│       ├── metadata.json
│       ├── metadata.pkl
│       └── setfit_model/               # Model weights & configuration
│
├── 📁 data/                            # Data Storage & Results
│   ├── input/                          # Source datasets & taxonomies
│   │   ├── training_dataset_classified.csv           # Main classified training data
│   │   ├── training_dataset_classified_2025-06-17.csv # Updated training classifications
│   │   ├── revised_taxonomy_final.json               # SetFit taxonomy structure
│   │   ├── parallel_classifications.json             # Individual record classifications
│   │   └── taxonomy*.json                            # Various taxonomy versions
│   │
│   ├── checkpoints/                    # Pipeline State Persistence
│   │   ├── pipeline_state.json         # Current pipeline stage status
│   │   ├── classification_checkpoint.pkl # Classification progress checkpoints
│   │   ├── hash_lookup.pkl             # PersonId → field hash mappings (enhanced with subject flags)
│   │   ├── string_dict.pkl             # Hash → original string mappings
│   │   ├── string_counts.pkl           # Hash → frequency count mappings
│   │   ├── field_hash_mapping.pkl      # Hash → field type relationships
│   │   ├── composite_subject_mapping.pkl # Composite → subject hash mappings
│   │   ├── processed_hashes.pkl        # Real-time embedding processing checkpoints
│   │   ├── batch_processed_hashes.pkl  # Batch embedding processing checkpoints
│   │   ├── batch_jobs.pkl              # OpenAI batch job tracking and metadata
│   │   ├── batch_queue_state.pkl       # Automated queue state (active, pending, completed batches)
│   │   ├── batch_blacklisted_files.pkl # Blacklisted batch files to avoid reprocessing
│   │   ├── batch_requests_*.jsonl      # JSONL files uploaded to OpenAI Batch API
│   │   ├── batch_results_*.jsonl       # JSONL results downloaded from OpenAI
│   │   ├── cache/                      # Performance optimization caches
│   │   │   └── imputation_cache.pkl    # Subject imputation results cache
│   │   └── *.pkl                       # Various stage-specific checkpoints
│   │
│   ├── output/                         # Results, Reports & Visualizations
│   │   ├── 📊 Performance Results
│   │   │   ├── test_results_filtered.csv            # Detailed prediction analysis (false positives)
│   │   │   ├── pipeline_metrics.json               # Overall performance metrics
│   │   │   ├── cluster_summary_report_*.json       # Clustering analysis results
│   │   │   ├── entity_matches.csv                  # Identified entity matches
│   │   │   ├── entity_clusters.json                # Transitive clustering results
│   │   │   ├── batch_embedding_metrics.json        # Batch processing performance metrics
│   │   │   └── subject_imputation_results_*.json   # Subject imputation detailed results
│   │   │
│   │   ├── 📈 Visualizations & Reports
│   │   │   ├── plots/                               # Feature analysis visualizations
│   │   │   │   ├── feature_distributions/           # Individual feature distribution plots
│   │   │   │   ├── class_separation/                # ROC/PR curves for each feature
│   │   │   │   ├── confusion_matrix.png
│   │   │   │   ├── feature_importance.png
│   │   │   │   └── probability_distribution.png
│   │   │   │
│   │   │   ├── reports/
│   │   │   │   ├── feature_visualization_report.html # Interactive feature analysis
│   │   │   │   └── subject_quality_audit_*.json     # Subject quality audit reports
│   │   │   │
│   │   │   └── viz/                                 # Pipeline diagrams
│   │   │       ├── pipeline.svg                     # Main pipeline architecture
│   │   │       └── taxonomy.svg                     # Taxonomy structure diagram
│   │   │
│   │   ├── 📋 Classification Results
│   │   │   ├── parallel_classifications.json        # Individual record classifications
│   │   │   ├── individual_classifications.json.partial # Partial classification results
│   │   │   └── updated_identity_classification_map_*.json # Identity mappings
│   │   │
│   │   └── 📊 Telemetry & Monitoring
│   │       └── telemetry/                           # Performance monitoring data
│   │           └── telemetry_*.json                 # Timestamped execution metrics
│   │
│   ├── ground_truth/                   # Training Data
│   │   └── labeled_matches.csv         # Ground truth entity pairs (left,right,match)
│   │
│   └── tmp/                           # Temporary Processing Files
│       ├── Yale_catalog_disambiguated_names*.csv    # Catalog processing intermediates
│       ├── clusters.csv
│       ├── missing.csv
│       └── *.csv                       # Various temporary datasets
│
├── 📁 logs/                            # Execution Logs
│   ├── pipeline.log                    # Main pipeline execution log
│   ├── individual_classification_verification.log          # Sequential classification log
│   └── individual_classification_verification_parallel.log # Parallel classification log
│
├── 📁 Data Extraction & Processing      # Pre-Pipeline Data Preparation
│   ├── extract-names-benchmark-2024-12-16-csv.xq   # XQuery BIBFRAME→CSV conversion
│   ├── extract-names-2024-12-16.xq                 # Original XQuery extraction script
│   └── fix-classifications.xq                      # Classification correction script
│
├── 📁 Analysis & Debugging Tools       # Development & Analysis Scripts
│   ├── analyze_false_positives.py      # False positive analysis
│   ├── inspect_test_data.py            # Test data inspection
│   ├── test_taxonomy_*.py              # Taxonomy feature testing
│   ├── debug_*.py                      # Various debugging utilities
│   ├── regenerate_*.py                 # Plot regeneration scripts
│   └── classification_embedding_script_*.py # Embedding analysis tools
│
├── 📁 Visualization & Presentation     # Documentation & Presentation
│   ├── create_presentation_visuals.py  # Presentation chart generation
│   ├── svg_to_png.py                   # Format conversion utility
│   ├── README.md                       # Main project documentation
│   ├── project_structure.md            # This file - detailed project structure
│   ├── Executive_Summary_Lightning_Talk.md # Project summary
│   ├── Lightning_Talk_Script.md        # Presentation script
│   ├── Real_Data_Examples_Summary.md   # Data examples documentation
│   └── *.md                           # Additional documentation files
│
└── 📁 Development Environment          # Development Setup
    ├── venv/                           # Python virtual environment
    ├── LICENSE                         # Project license
    └── .env                           # Environment variables (API keys)
```

## Key Data Formats

### Input Data Structure
```csv
composite,person,roles,title,provision,subjects,personId,setfit_prediction,is_parent_category
"Contributor: Bach, Johann Sebastian, 1685-1750...",Bach\, Johann Sebastian,Contributor,The Well-Tempered Clavier,...,...,...,12345#Agent700-1,Music\, Sound\, and Sonic Arts,FALSE
```

### Ground Truth Format
```csv
left,right,match
16044091#Agent700-32,9356808#Agent100-11,true
16044091#Agent700-32,9940747#Hub240-13-Agent,true
```

### Pipeline Output
- **Entity Matches**: CSV with predicted pairs and confidence scores
- **Entity Clusters**: JSON with transitive clustering results  
- **Classification Results**: JSON with individual record classifications
- **Performance Metrics**: Detailed confusion matrix and feature analysis

## Feature Engineering System

### Currently Active Features (5 total)
1. **person_cosine**: Cosine similarity between person name embeddings
2. **person_title_squared**: Squared person-title interaction term
3. **composite_cosine**: Full record composite similarity
4. **taxonomy_dissimilarity**: SetFit domain classification differences
5. **birth_death_match**: Binary temporal matching with tolerance

### Feature Scaling Strategy
- **Feature Groups**: Domain-specific scaling (person: 98th percentile, title: 95th, context: 90th)
- **Binary Preservation**: Maintains exact 0.0/1.0 values for binary indicators
- **Training/Production Consistency**: Identical scaling in both environments

## Technology Stack

### Core Dependencies
- **Vector Database**: Weaviate with HNSW indexing
- **Embeddings**: OpenAI text-embedding-3-small (1536D)
- **ML Framework**: Custom logistic regression with gradient descent
- **Taxonomy Classification**: SetFit (Sentence Transformers + logistic head)
- **Parallel Processing**: asyncio/aiohttp for API rate limit optimization

### Production Features
- **Checkpointing**: Complete pipeline state persistence and resumption
- **Rate Limiting**: Anthropic API optimization for high-tier accounts (4,000 RPM)
- **Error Resilience**: Comprehensive exception handling and retry logic
- **Monitoring**: Detailed telemetry, memory usage tracking, and performance metrics
- **Scalability**: Configurable batch processing and worker allocation
- **Environment Adaptation**: Local vs. production resource allocation (4-64 cores, 16GB-247GB RAM)
- **Subject Enhancement**: Automated quality audit and missing value imputation
- **Production-Ready Automated Batch Processing**: Self-managing 16-batch queue with bulletproof quota management, one-at-a-time submission with verification, automatic failed job cleanup, and comprehensive error recovery

## Performance Characteristics

### Latest Results (Test Set: 14,930 pairs)
- **Precision**: 99.55% (9,955 TP, 45 FP)
- **Recall**: 82.48% (2,114 FN, 9,955 TP)
- **F1-Score**: 90.22%
- **Specificity**: 98.43%
- **Accuracy**: 85.54%

### Computational Efficiency
- **ANN Reduction**: 99.23% reduction in pairwise comparisons via vector similarity
- **Cluster Analysis**: 163 clusters, 4,672 entities, 83,921 comparisons
- **Throughput**: Optimized for high-volume library catalog processing

## Development Workflow

### Pipeline Execution
```bash
# Complete pipeline with real-time embeddings
python main.py --config config.yml

# Complete pipeline with fully automated batch embeddings (50% cost savings)
# Set use_batch_embeddings: true and use_automated_queue: true in config.yml
python main.py --config config.yml

# Environment-specific execution
PIPELINE_ENV=prod python main.py --config config.yml  # Production settings
PIPELINE_ENV=local python main.py --config config.yml # Local settings (default)

# Stage-specific execution  
python main.py --start training --end classification

# Manual batch processing commands
python main.py --batch-status       # Check batch job status
python main.py --batch-results      # Download and process results

# Resume from checkpoints
python main.py --resume

# Reset specific stages
python main.py --reset training classifying
```

### Batch Processing Management
```bash
# Fully automated batch processing (recommended)
python batch_manager.py --create      # Starts self-managing automated queue system

# Traditional manual batch operations (if automation disabled)
python batch_manager.py --status      # Check job status
python batch_manager.py --download    # Process results

# Advanced batch operations
python main.py --batch-status          # Check all batch job statuses
python main.py --batch-results         # Download and process completed results
python batch_manager.py --recover      # Recover jobs from OpenAI API
python batch_manager.py --investigate  # Investigate failed batch jobs
python batch_manager.py --resubmit     # Resubmit failed jobs
python batch_manager.py --cancel       # Cancel uncompleted jobs (frees quota)
python batch_manager.py --reset        # Reset embedding stage (clear checkpoints)
```

### Individual Record Classification
```bash
# Parallel processing (optimized for API limits)
python scripts/verify_individual_classifications_parallel.py \
    --csv data/input/training_dataset.csv \
    --taxonomy data/input/revised_taxonomy_final.json \
    --output data/output/parallel_classifications.json \
    --concurrency 5
```

### SetFit Training
```bash
# Train hierarchy-aware taxonomy classifier
python setfit/train_setfit_classifier.py \
    --csv_path data/input/training_dataset.csv \
    --ground_truth_path data/output/updated_identity_classification_map_v6_pruned.json \
    --output_dir ./setfit_model_output
```

### Subject Enhancement Operations
```bash
# Manual subject quality audit
python main.py --start subject_quality --end subject_quality

# Manual subject imputation
python main.py --start subject_imputation --end subject_imputation

# Combined subject enhancement
python main.py --start subject_quality --end subject_imputation

# Environment-specific subject enhancement
PIPELINE_ENV=prod python main.py --start subject_quality --end subject_imputation
```

## Recent Developments

### Subject Enhancement System (NEW)
- **Quality Audit Module**: Automated evaluation of existing subject field quality using composite field vector similarity
- **Imputation Module**: Missing subject field population using weighted centroid calculation from similar composite fields
- **Vector-Based Join Strategy**: Leverages semantic similarity to identify appropriate subject values
- **Automated Remediation**: High-confidence quality improvements with configurable thresholds
- **Performance Caching**: Imputation results caching with configurable size limits (10,000 entries)
- **Comprehensive Reporting**: Detailed audit reports with statistics and remediation metrics

### Enhanced Configuration Management
- **Environment-Specific Settings**: Automatic local vs. production resource allocation
- **Advanced Resource Tuning**: 64-core production optimization (32 workers, 5000-request batches)
- **Weaviate Optimization**: Environment-specific HNSW parameters and connection pooling
- **Feature Group Scaling**: Domain-specific scaling strategies with percentile normalization
- **Cluster Validation**: Parallel validation with 48 workers for production environments

### Fully Automated Batch Processing System (PRODUCTION-READY)
- **Self-Managing Queue**: Automatically maintains exactly 16 active batches with zero manual intervention
- **Bulletproof Quota Management**: 
  - Accurately counts ALL requests (including failed jobs) toward OpenAI's 1M limit
  - Real-time quota probing with actual API test submissions for verification
  - Conservative 800K request limit (80% of OpenAI's 1M) with intelligent safety margins
- **One-at-a-Time Submission**: Conservative batch submission with real-time verification between each
- **Intelligent Recovery System**:
  - Graceful quota exceeded detection with immediate error recognition
  - 32-hour polling system: checks every 30 minutes for quota recovery
  - Automatic position preservation and resumption from exact failure point
  - Progressive wait strategies based on natural job completion (no manual cleanup needed)
- **Real-time Verification**: 
  - Immediate batch status validation after each submission
  - Pre-submission quota checks (stops at 95% usage)
  - Post-submission quota verification (stops at 90% usage)
- **Enhanced Error Handling**:
  - Consecutive error tracking with progressive delays
  - Network timeout categorization and intelligent retry
  - Graceful degradation with state preservation
- **Complete Automation**: Runs until all work is complete with sophisticated monitoring
- **Advanced Monitoring**: 
  - Real-time quota usage percentages and trends
  - Queue state tracking with cycle-by-cycle logging
  - Detailed error categorization and recovery metrics
- **Batch API Integration**: OpenAI Batch API support with 50% cost savings and 24-hour turnaround
- **State Persistence**: Complete queue state recovery from interruptions with comprehensive checkpoint management

### Individual Record Classification Enhancement
- **Parallel API Processing**: Optimized for Anthropic rate limits (200K tokens/min)
- **Rate Limiting**: Automatic token usage analysis and concurrency adjustment
- **Progress Logging**: Detailed classification progress tracking
- **Compatibility**: Updated training scripts to use personId instead of identity mapping

### Pipeline Improvements
- **Feature Analysis**: Identified inactive features (taxonomy_dissimilarity, birth_death_match)
- **Performance Optimization**: Reduced ANN search overhead by 99.23%
- **Error Analysis**: Comprehensive false positive pattern analysis
- **Visualization**: Enhanced feature importance and distribution plots

## Configuration System

### Environment-Specific Resource Allocation

The system automatically adapts resource allocation based on the `PIPELINE_ENV` environment variable:

#### Local Development Configuration (default)
```yaml
local_resources:
  preprocessing_workers: 4
  preprocessing_batch_size: 50
  embedding_workers: 4
  embedding_batch_size: 32
  feature_workers: 4
  classification_workers: 8
```

#### Production Configuration (64 cores, 247GB RAM)
```yaml
prod_resources:
  preprocessing_workers: 32               # Utilize half cores for I/O bound preprocessing
  preprocessing_batch_size: 500          # Larger batches for better memory utilization
  embedding_workers: 16                  # Balance between API rate limits and parallelism
  embedding_batch_size: 100              # Larger batches to reduce API overhead
  feature_workers: 48                    # High parallelism for CPU-intensive computation
  classification_workers: 32             # High parallelism for classification
```

### Subject Enhancement Configuration

#### Subject Quality Audit Settings
```yaml
subject_quality_audit:
  enabled: true                      # Enable automatic subject quality audit
  similarity_threshold: 0.70         # Minimum composite similarity for alternatives
  remediation_threshold: 0.60        # Quality score threshold for remediation
  min_alternatives: 3                # Minimum alternative subjects required
  max_candidates: 100                # Maximum candidate composites to consider
  frequency_weight: 0.3              # Weight for subject frequency in scoring
  similarity_weight: 0.7             # Weight for vector similarity in scoring
  auto_remediate: true               # Automatically apply high-confidence improvements
  confidence_threshold: 0.80         # Confidence threshold for automatic remediation
```

#### Subject Imputation Settings
```yaml
subject_imputation:
  enabled: true                      # Enable automatic subject imputation
  similarity_threshold: 0.65         # Minimum composite similarity for candidates
  confidence_threshold: 0.70         # Confidence threshold for applying imputed subjects
  min_candidates: 3                  # Minimum candidate subjects required
  max_candidates: 150                # Maximum candidate composites to consider
  frequency_weight: 0.3              # Weight for subject frequency in scoring
  centroid_weight: 0.7               # Weight for centroid similarity in scoring
  use_caching: true                  # Enable caching for performance
  cache_size_limit: 10000            # Maximum cached results
```

### Batch Processing Configuration

#### OpenAI Batch API Settings
```yaml
use_batch_embeddings: true          # Enable OpenAI Batch API (50% cost savings)
batch_embedding_size: 50000         # Number of requests per batch file
max_requests_per_file: 50000        # Maximum requests per JSONL file
batch_manual_polling: true          # Manual polling (recommended)
batch_poll_interval: 300            # Seconds between status polls (auto mode)
batch_max_wait_time: 86400          # Maximum wait time (24 hours)

# Automated Queue Management Configuration
use_automated_queue: true           # Enable automated 16-batch queue system
max_active_batches: 16              # Maximum concurrent batches (default: 16)  
queue_poll_interval: 1800           # 30 minutes between status checks (default: 1800)
request_quota_limit: 800000         # Conservative 800K request limit (80% of 1M)

# Traditional quota management (fallback)
token_quota_limit: 500000000        # 500M token limit
max_concurrent_jobs: 50             # Limit concurrent batch jobs
quota_safety_margin: 0.1            # 10% safety margin
```

### Weaviate Environment-Specific Tuning

#### Local Development Settings
```yaml
local_weaviate:
  weaviate_batch_size: 100
  weaviate_ef: 128
  weaviate_max_connections: 64
  weaviate_connection_pool_size: 16
  weaviate_query_concurrent_limit: 4
```

#### Production Settings (64 cores, 247GB RAM)
```yaml
prod_weaviate:
  weaviate_batch_size: 1000           # Much larger batch sizes
  weaviate_ef: 256                    # Higher EF for better recall
  weaviate_max_connections: 128       # More connections for parallelism
  weaviate_connection_pool_size: 64   # Match available cores
  weaviate_query_concurrent_limit: 32 # High concurrency for production
```

## 🎯 Current Status & Future TODOs

### ✅ **COMPLETED: Production-Ready System**
- **Core Pipeline**: Fully functional with 99.55% precision entity resolution
- **Automated Batch Processing**: Bulletproof quota management with zero manual intervention
- **Subject Enhancement**: Quality audit and imputation with vector similarity
- **Environment Adaptation**: Local vs. production resource allocation
- **Comprehensive Error Handling**: Robust recovery and state persistence

### 🔧 **TODO: Feature Analysis & Optimization** 
- **Feature Performance Analysis**: 
  - `taxonomy_dissimilarity`: Working correctly (returns 0.0 when domains match, >0 when different)
  - `birth_death_match`: Working correctly (returns 0.0 when no temporal data, 1.0 when matching)
  - **Analysis Needed**: Investigate why these features show low variance in current dataset
  - **Impact**: Understanding feature distributions could guide data enhancement or feature engineering improvements
- **Performance Optimization**:
  - Investigate ANN search parameter tuning for the remaining 0.77% comparison overhead
  - Consider advanced clustering algorithms for improved entity grouping
- **API Integration Enhancements**:
  - Add support for OpenAI's newer embedding models as they become available
  - Implement adaptive batch sizing based on real-time quota availability

### 📊 **TODO: Analytics & Monitoring**
- **Advanced Telemetry**: 
  - Cost tracking dashboard with real-time API usage analysis
  - Performance trend analysis across different dataset characteristics
- **Quality Metrics**:
  - Automated false positive pattern detection and alerting
  - Subject enhancement effectiveness tracking with before/after comparisons

### 🚀 **TODO: Scalability Enhancements**
- **Distributed Processing**: 
  - Multi-node Weaviate cluster support for datasets beyond current scale
  - Parallel batch queue management across multiple OpenAI accounts
- **Advanced Configuration**:
  - Dynamic resource allocation based on system load
  - Automatic parameter tuning based on dataset characteristics

### 🔐 **TODO: Security & Compliance**
- **Data Protection**: 
  - Enhanced encryption for checkpoint files containing sensitive data
  - Audit trail for all data processing and entity resolution decisions
- **API Security**:
  - API key rotation and secure credential management
  - Request rate monitoring and anomaly detection

---

**This structure represents a production-ready entity resolution system with comprehensive tooling for analysis, debugging, and performance optimization. The automated batch processing system is now bulletproof and ready for large-scale deployment.**