# Entity Resolution Pipeline: System Architecture

## System Overview

The Entity Resolution Pipeline is a production-ready system designed to identify and resolve entities across MARC 21 records in the Yale University Library Catalog. The pipeline processes library catalog CSV data, generates OpenAI vector embeddings, calculates similarity features, and uses logistic regression classification to identify matching entities with high precision.

**Latest Performance Metrics** (Test Results 2025-06-05):
- **Precision: 99.55%** (9,955 true positives, 45 false positives) - extremely low false positive rate
- **Recall: 82.48%** (2,114 false negatives out of 12,069 actual matches)
- **F1 Score: 90.22%**, **Accuracy: 85.54%** on 14,930 test pairs
- **Specificity: 98.43%** (strong negative class precision)

**Architecture Highlights**:
- **Dual Classification System**: Main logistic regression pipeline + SetFit hierarchical taxonomy classifier
- **Feature Engineering**: 5 similarity features with domain-specific scaling
- **Production-Ready**: Checkpointing, resumption, telemetry, error resilience, memory management
- **Vector Database**: Weaviate integration with hash-based deduplication

## Core Pipeline Architecture

The system implements a five-stage pipeline orchestrated by `main.py` through `src/orchestrating.py`:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Preprocessing  │────>│  Embedding &    │────>│  Training       │────>│  Classification │────>│  Reporting      │
│  (MARC parsing) │     │  Indexing       │     │  (LogReg + GD)  │     │  (Entity Match) │     │  (HTML/CSV)     │
│  • CSV input   │     │  • OpenAI API   │     │  • Feature      │     │  • Clustering   │     │  • Dashboards   │
│  • Deduplication│     │  • Weaviate DB  │     │    engineering  │     │  • Threshold    │     │  • Metrics      │
│  • Hash lookup  │     │  • Vector index │     │  • Scaling      │     │    application  │     │  • Exports      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Entry Point and Orchestration

* **main.py**: CLI entry point with comprehensive argument parsing
  * Command options: `--start STAGE`, `--end STAGE`, `--reset [STAGES]`, `--resume`, `--status`, `--disable-scaling`
  * Configuration loading from `config.yml`
  * Memory monitoring and logging setup
  * Docker integration for Weaviate (`--docker` flag)

* **src/orchestrating.py**: `PipelineOrchestrator` class managing complete workflow
  * Five-stage pipeline with checkpointing and resumption
  * Error handling with stage-by-stage fallback
  * Metrics collection and JSON export
  * Legacy stage support (deprecated `embedding` and `indexing` stages)

## Pipeline Stages Deep Dive

### 1. Preprocessing (`src/preprocessing.py`)

**Input**: CSV files with MARC 21 library catalog records
**Core Functions**: `process_data()`, `process_file()`, `hash_string()`, `create_string_dict()`

**Data Structure**:
```csv
composite,person,roles,title,provision,subjects,personId,setfit_prediction,is_parent_category
"Contributor: Allen, William...",Allen\, William,Contributor,Dēmosthenous Logoi...,Oxonii...,,...,2117946#Agent700-25,Humanities\, Thought\, and Interpretation,TRUE
```

**Key Operations**:
- MD5-based deduplication: `hashlib.md5(string.encode('utf-8')).hexdigest()`
- Hash lookup creation: Maps `personId` → field hashes
- String dictionary: Maps hash → original string value
- Field mapping: Tracks hash → field type relationships
- Frequency analysis for string occurrence counts

**Outputs**: `hash_lookup.pkl`, `string_dict.pkl`, `field_hash_mapping.pkl`, `string_counts.pkl`

### 2. Embedding & Indexing (`src/embedding_and_indexing.py`)

**Core Class**: `EmbeddingAndIndexingPipeline`
**API Integration**: OpenAI `text-embedding-3-small` (1536 dimensions)
**Vector Database**: Weaviate with HNSW indexing

**Weaviate Schema**:
```python
{
  "original_string": TEXT,    # Original string value
  "hash_value": TEXT,         # MD5 hash for deduplication 
  "field_type": TEXT,         # Field category (person, title, etc.)
  "frequency": INT            # Occurrence count for weighting
}
```

**Vector Configuration**:
- Distance metric: Cosine similarity
- Index type: HNSW (Hierarchical Navigable Small World)
- Parameters: `ef=128`, `maxConnections=64`, `efConstruction=128`
- UUID generation: Deterministic UUID5 from `hash_value` + `field_type`

**Processing**:
- Batch embedding generation (configurable batch size: 32)
- Rate limiting and retry logic with exponential backoff
- Direct indexing (no intermediate vector storage)
- Checkpoint-based resumption via `processed_hashes.pkl`

### 3. Training (`src/training.py`)

**Core Class**: `EntityClassifier`
**Algorithm**: Logistic regression with gradient descent
**Data Source**: Ground truth pairs from `data/ground_truth/labeled_matches.csv`

**Model Configuration**:
- Learning rate: 0.01 (configurable)
- Max iterations: 1000
- Batch size: 256
- L2 regularization: λ=0.01
- Class weighting: 5:1 (positive:negative)
- Decision threshold: 0.5

**Training Process**:
1. Load labeled pairs: `left,right,match` format
2. Feature engineering via `FeatureEngineering` class
3. Feature scaling via `ScalingBridge` → `LibraryCatalogScaler`
4. Mini-batch gradient descent with deterministic shuffling
5. Early stopping with validation monitoring
6. Model serialization to `classifier_model.pkl`

**Features Currently Enabled**:
- `person_cosine`: Cosine similarity between person name embeddings
- `person_title_squared`: Squared person-title interaction term  
- `composite_cosine`: Full record composite similarity
- `taxonomy_dissimilarity`: SetFit domain classification dissimilarity
- `birth_death_match`: Binary birth/death year matching with tolerance

### 4. Classification (`src/classifying.py`)

**Core Class**: `EntityClassification`
**Process**: Apply trained classifier to identify entity matches across full dataset

**Key Features**:
- Batch processing with configurable batch size (500)
- Memory management and garbage collection
- Telemetry collection with detailed metrics
- Thread-safe caching and error resilience
- Consistent feature scaling matching training environment

**Output Generation**:
1. **Entity Matches**: CSV with predicted match pairs and confidence scores
2. **Entity Clusters**: JSON with transitive clustering results
3. **Detailed Results**: Include both raw and normalized feature values
4. **Telemetry**: Performance metrics and error tracking

### 5. Reporting (`src/reporting.py`)

**Core Functions**: `generate_report()`, `generate_detailed_test_results()`
**Core Class**: `FeatureOptimizationReporter`

**Reporting Capabilities**:
- **Detailed Test Analysis**: `detailed_test_results_{timestamp}.csv` with raw/normalized features, confidence scores, TP/FP/TN/FN classification
- **Interactive HTML Dashboards**: Parameter correlation analysis and configuration optimization results
- **Diagnostic Analysis**: Automatic detection of problematic binary indicator values (e.g., identical strings with incorrect indicators)
- **Feature Importance Visualization**: `feature_importance_{timestamp}.png` using actual model weights
- **Structured Logging**: JSONL format for configuration optimization tracking
- **Multi-format Output**: CSV, JSON, JSONL, HTML, PNG visualizations
- **Performance Metrics**: Comprehensive confusion matrix analysis with specificity, NPV, accuracy
- **Thread-safe Operations**: File locking for concurrent report generation

**Output Files**:
- `detailed_test_results_{timestamp}.csv` - Complete prediction analysis
- `test_summary_{timestamp}.json` - Performance metrics summary  
- `problematic_indicators_{timestamp}.json` - Binary feature diagnostics
- `feature_importance_{timestamp}.png` - Model weight visualizations
- `configuration_dashboard.html` - Interactive optimization dashboard
- `configuration_results.jsonl` - Structured configuration logs

## Feature Engineering System

### Core Class: `FeatureEngineering` (`src/feature_engineering.py`)

**Architecture Highlights**:
- Thread-safe versioned caching system (`VersionedCache`)
- Feature substitution mechanism for custom implementations
- Binary vs continuous similarity metric options
- Deterministic processing with seeded randomization
- Advanced error handling and diagnostic logging

**Feature Categories**:

1. **String Similarity Features**:
   - Levenshtein distance with binary thresholding
   - Jaro-Winkler similarity with configurable parameters
   - Normalized string preprocessing with Unicode handling

2. **Vector Similarity Features**:
   - Cosine similarity between OpenAI embeddings
   - Squared interaction terms (e.g., `person_title_squared`)
   - Fallback mechanisms for missing vectors

3. **Domain-Specific Features**:
   - `birth_death_match`: Sophisticated temporal matching with tolerance
   - `taxonomy_dissimilarity`: SetFit classification differences
   - Role-adjusted similarity calculations

4. **Composite Features**:
   - Full record similarity (`composite_cosine`)
   - Multi-field interaction terms
   - Weighted combinations based on field importance

### Scaling Architecture

**Core Classes**: `ScalingBridge` (`src/scaling_bridge.py`), `LibraryCatalogScaler` (`src/robust_scaler.py`)

**Feature Group Treatment**:
```python
feature_groups = {
    "person_features": ["person_cosine"],           # 98th percentile
    "title_features": ["person_title_squared"],     # 95th percentile  
    "context_features": ["composite_cosine"],       # 90th percentile
    "binary_features": ["birth_death_match"]        # No scaling (preserve 0.0/1.0)
}
```

**Critical Features**:
- Identical scaling between training and production environments
- Binary feature preservation (exact 0.0 or 1.0 values)
- Domain-specific percentile thresholds
- Robust error handling and fallback mechanisms

## Dual Classification Architecture

### 1. Main Pipeline: Entity Resolution
- **Purpose**: Identify matching person entities across library records
- **Algorithm**: Logistic regression with feature engineering
- **Performance**: 99.55% precision, 82.48% recall

### 2. SetFit System: Hierarchical Taxonomy Classification

**Location**: `/setfit/` directory
**Purpose**: Classify entities into hierarchical domain categories
**Algorithm**: SetFit (Sentence Transformers + logistic head)

**Key Features**:
- Handles extreme class imbalance by mapping rare classes (< 8 examples) to parent categories
- GPU support (CUDA, MPS, CPU fallback)
- Hierarchical accuracy vs original accuracy reporting
- Production-ready with confidence handling

**Training Command**:
```bash
python setfit/train_setfit_classifier.py \
    --csv_path data/input/training_dataset.csv \
    --ground_truth_path data/output/updated_identity_classification_map_v6_pruned.json \
    --output_dir ./setfit_model_output
```

**Integration**: Provides `setfit_prediction` field for taxonomy dissimilarity feature

## Configuration Management

### Primary Configuration: `config.yml`

**Key Sections**:
```yaml
# Deterministic processing
random_seed: 42

# Resource allocation
preprocessing_workers: 4
embedding_batch_size: 32
classification_workers: 8
classification_batch_size: 500

# API configuration  
embedding_model: "text-embedding-3-small"
embedding_dimensions: 1536

# Feature configuration
features:
  enabled: ["person_cosine", "person_title_squared", "composite_cosine", 
           "taxonomy_dissimilarity", "birth_death_match"]
```

### Scaling Configuration: `scaling_config.yml`
- Feature group definitions and percentile thresholds
- Binary feature preservation rules
- Domain-specific scaling parameters

## Production Features

### Checkpointing and Resumption
- Persistent state management via `src/checkpoint_manager.py`
- Stage-level checkpoints with metadata
- Resume capability: `python main.py --resume`
- Status monitoring: `python main.py --status`

### Error Resilience
- Comprehensive exception handling
- Retry logic with exponential backoff (Tenacity library)
- Graceful degradation for missing data
- Memory management and cleanup

### Monitoring and Telemetry
- Detailed performance metrics collection
- Memory usage monitoring (psutil)
- Progress tracking with tqdm
- Comprehensive logging with configurable levels

### Development and Debugging
- Feature-level debugging modes
- Deterministic processing guarantees
- Cache statistics and hit rate monitoring
- Transaction ID tracking for multi-threaded debugging

## Data Flow and Dependencies

```
Input CSV (MARC 21 records)
    ↓
Preprocessing: Hash-based deduplication
    ↓
String Dict + Hash Lookup + Field Mapping
    ↓
Embedding & Indexing: OpenAI API → Weaviate
    ↓
Vector Database (EntityString collection)
    ↓
Training: Feature Engineering + Logistic Regression
    ↓
Trained Classifier Model (classifier_model.pkl)
    ↓
Classification: Entity Matching + Clustering
    ↓
Results: entity_matches.csv + entity_clusters.json + HTML Reports
```

## Key Dependencies

**External Services**:
- OpenAI API (text-embedding-3-small)
- Weaviate vector database (Docker)

**Python Libraries**:
- `weaviate-client` (v4 API)
- `openai`, `numpy`, `scikit-learn`
- `tenacity` (retry logic)
- `psutil` (monitoring)
- `tqdm` (progress bars)

**Optional Extensions**:
- SetFit ecosystem (`setfit/`)
- Custom feature implementations
- Alternative scaling strategies

This architecture demonstrates production-quality entity resolution with feature engineering, robust error handling, and scalable vector operations.