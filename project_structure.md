# Entity Resolution Pipeline: Complete Project Structure

## Overview

This entity resolution pipeline processes Yale University Library Catalog data (derived from MARC 21 records) to identify and resolve entity matches across catalog entries. The system combines vector embeddings, feature engineering, and machine learning to achieve **99.55% precision** and **82.48% recall** in production.

## Pipeline Architecture

The system implements a **dual classification approach**:

1. **Main Pipeline**: Entity resolution using logistic regression with engineered features
2. **Auxiliary Classification**: Individual record taxonomy classification using parallel API processing and SetFit models

### Core Pipeline Stages

```
Input CSV → Preprocessing → Embedding & Indexing → Training → Classification → Reporting
     ↓            ↓              ↓              ↓            ↓            ↓
  Hash-based   OpenAI API    Feature Eng.   Similarity   Clustering   HTML/CSV
  Dedup        Weaviate DB   Scaling        Matching     Results      Reports
```

## Project Directory Structure

```
entity_resolver/
├── 📁 Core Pipeline Components
│   ├── main.py                          # CLI entry point with stage control
│   ├── config.yml                       # Main pipeline configuration
│   ├── scaling_config.yml               # Feature scaling configuration
│   ├── docker-compose.yml               # Weaviate vector database setup
│   └── requirements.txt                 # Python dependencies
│
├── 📁 src/                              # Core pipeline modules
│   ├── orchestrating.py                # Pipeline orchestration & stage management
│   ├── preprocessing.py                # CSV processing & hash-based deduplication
│   ├── embedding_and_indexing.py       # OpenAI embeddings & Weaviate integration
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
│   │   ├── hash_lookup.pkl             # PersonId → field hash mappings
│   │   ├── string_dict.pkl             # Hash → original string mappings
│   │   ├── field_hash_mapping.pkl      # Hash → field type relationships
│   │   ├── processed_hashes.pkl        # Embedding processing checkpoints
│   │   └── *.pkl                       # Various stage-specific checkpoints
│   │
│   ├── output/                         # Results, Reports & Visualizations
│   │   ├── 📊 Performance Results
│   │   │   ├── test_results_filtered.csv            # Detailed prediction analysis (false positives)
│   │   │   ├── pipeline_metrics.json               # Overall performance metrics
│   │   │   ├── cluster_summary_report_*.json       # Clustering analysis results
│   │   │   ├── entity_matches.csv                  # Identified entity matches
│   │   │   └── entity_clusters.json                # Transitive clustering results
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
│   │   │   │   └── feature_visualization_report.html # Interactive feature analysis
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
# Complete pipeline
python main.py --config config.yml

# Stage-specific execution  
python main.py --start training --end classification

# Resume from checkpoints
python main.py --resume

# Reset specific stages
python main.py --reset training classifying
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

## Recent Developments

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

This structure represents a production-ready entity resolution system with comprehensive tooling for analysis, debugging, and performance optimization.