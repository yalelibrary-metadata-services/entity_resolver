# Entity Resolution Pipeline

A production-ready system for identifying and resolving entities across MARC 21 records in the Yale University Library Catalog. The pipeline achieves **99.55% precision** and **82.48% recall** using feature engineering and logistic regression classification.

## Key Features

- **High Performance**: 99.55% precision, 90.22% F1-score on real library catalog data
- **Dual Classification**: Main entity resolution + SetFit hierarchical taxonomy classification  
- **Production Ready**: Checkpointing, resumption, telemetry, error resilience
- **Vector Database**: Weaviate integration with OpenAI embeddings
- **Feature Engineering**: 5 similarity features with domain-specific scaling

## Quick Start

### Prerequisites

- Python 3.10+ 
- Docker and Docker Compose (for Weaviate)
- OpenAI API key

### Installation

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd entity_resolver
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Start Weaviate vector database**:
   ```bash
   docker-compose up -d weaviate
   # Wait ~30 seconds for startup
   curl http://localhost:8080/v1/.well-known/ready  # Check readiness
   ```

### Basic Usage

**Run complete pipeline**:
```bash
python main.py --config config.yml
```

**Run specific stages**:
```bash
# Just preprocessing
python main.py --start preprocessing --end preprocessing

# Training only  
python main.py --start training --end training

# Resume from last completed stage
python main.py --resume

# Check pipeline status
python main.py --status
```

**Stage control**:
```bash
# Reset and re-run specific stages
python main.py --reset embedding_and_indexing training --start embedding_and_indexing

# Run with Docker Weaviate startup
python main.py --docker
```

## Data Extraction (Pre-Pipeline)

Before running the main pipeline, training datasets are generated using XQuery extraction from BIBFRAME catalog data.

### BIBFRAME to CSV Conversion

The entity resolution pipeline operates on CSV datasets extracted from Yale University Library's BIBFRAME catalog using:

**XQuery Script**: `extract-names-benchmark-2024-12-16-csv.xq`
- **Input**: BIBFRAME RDF/XML (converted from MARC21 catalog records)
- **Query Engine**: BaseX database with XQuery 4.0
- **Output**: CSV datasets with composite text fields optimized for embedding

**Key Processing Steps**:
1. **Authority Control**: Filters for Library of Congress authority records (excludes non-LC except FAST, SWD, GND)
2. **Entity Extraction**: Processes both Contributors (creators/editors) and Subjects (people as topics)
3. **Metadata Assembly**: Combines person names, roles, titles, subjects, and publication details
4. **Composite Text Generation**: Creates multi-line text fields optimized for semantic embedding

**Sample BIBFRAME Input**:
```xml
<bf:Work>
  <bf:contribution>
    <bf:Contribution>
      <bf:agent>
        <bf:Agent rdf:type="http://id.loc.gov/ontologies/bibframe/Person">
          <rdfs:label>Bach, Johann Sebastian, 1685-1750</rdfs:label>
          <bflc:marcKey>1001 $aBach, Johann Sebastian,$d1685-1750.</bflc:marcKey>
        </bf:Agent>
      </bf:agent>
      <bf:role rdf:resource="http://id.loc.gov/vocabulary/relators/cmp"/>
    </bf:Contribution>
  </bf:contribution>
</bf:Work>
```

**Generated CSV Output**:
```csv
composite,person,roles,title,attribution,provision,subjects,genres,relatedWork,recordId,personId
"Contributor: Bach, Johann Sebastian, 1685-1750
Title: The Well-Tempered Clavier
Attribution: edited by Johann Sebastian Bach
Subjects: Keyboard music; Fugues
Genres: Musical scores
Provision information: Leipzig: Breitkopf & Härtel, 1985","Bach, Johann Sebastian, 1685-1750",Composer,The Well-Tempered Clavier,edited by Johann Sebastian Bach,"Leipzig: Breitkopf & Härtel, 1985","Keyboard music; Fugues",Musical scores,,12345,12345#Agent100-1
```

**Note**: This data extraction step is performed separately from the main pipeline and generates the input CSV files used in Stage 1 (Preprocessing).

## Architecture Overview

The pipeline implements a **five-stage architecture**:

```
Input (MARC CSV) → Preprocessing → Embedding & Indexing → Training → Classification → Reporting
```

### Pipeline Stages

1. **Preprocessing** (`src/preprocessing.py`)
   - Parses MARC 21 CSV records
   - MD5-based deduplication and hash lookup creation
   - String frequency analysis

2. **Embedding & Indexing** (`src/embedding_and_indexing.py`)  
   - OpenAI `text-embedding-3-small` (1536D) embeddings
   - Weaviate vector database with HNSW indexing
   - Batch processing with rate limiting

3. **Training** (`src/training.py`)
   - Logistic regression with gradient descent
   - Feature engineering (5 features)
   - Class weighting (5:1) and L2 regularization

4. **Classification** (`src/classifying.py`)
   - Entity matching across full dataset
   - Transitive clustering and confidence scoring
   - Batch processing with telemetry

5. **Reporting** (`src/reporting.py`)
   - Interactive HTML dashboards with parameter correlation analysis
   - Detailed CSV exports with raw/normalized features and diagnostics
   - Feature importance visualizations and structured logging
   - Diagnostics for binary indicator anomalies

## Data Format

### Input CSV Structure
```csv
composite,person,roles,title,provision,subjects,personId,setfit_prediction,is_parent_category
"Contributor: Bach, Johann Sebastian...",Bach\, Johann Sebastian,Contributor,The Well-Tempered Clavier,...,...,...,12345#Agent700-1,Music and Sound Arts,FALSE
```

### Ground Truth Format
```csv
left,right,match
16044091#Agent700-32,9356808#Agent100-11,true
16044091#Agent700-32,9940747#Hub240-13-Agent,true
```

## Feature Engineering

### Feature Engineering

**Currently enabled features**:
- `person_cosine`: Cosine similarity between person name embeddings
- `person_title_squared`: Squared person-title interaction term
- `composite_cosine`: Full record composite similarity  
- `taxonomy_dissimilarity`: SetFit domain classification dissimilarity
- `birth_death_match`: Binary birth/death year matching with tolerance

### Scaling System

The pipeline includes feature scaling with:
- **Feature groups**: Different scaling for person, title, context, and binary features
- **Percentile-based**: 90th-98th percentile normalization by feature type
- **Binary preservation**: Maintains exact 0.0/1.0 values for binary indicators
- **Training/production consistency**: Identical scaling in both environments

### SetFit Integration

**Hierarchical taxonomy classification** (`setfit/` directory):
- Handles extreme class imbalance (< 8 examples → parent mapping)  
- GPU support (CUDA, MPS, CPU)
- Used for `taxonomy_dissimilarity` feature

**Training SetFit classifier**:
```bash
python setfit/train_setfit_classifier.py \
    --csv_path data/input/training_dataset.csv \
    --ground_truth_path data/output/updated_identity_classification_map_v6_pruned.json \
    --output_dir ./setfit_model_output
```

## Configuration

### Main Configuration (`config.yml`)

Key settings:
```yaml
# Deterministic processing
random_seed: 42

# Resource allocation  
preprocessing_workers: 4
embedding_batch_size: 32
classification_workers: 8
classification_batch_size: 500

# OpenAI configuration
embedding_model: "text-embedding-3-small"
embedding_dimensions: 1536

# Enabled features
features:
  enabled: ["person_cosine", "person_title_squared", "composite_cosine",
           "taxonomy_dissimilarity", "birth_death_match"]
```

### Scaling Configuration (`scaling_config.yml`)

Feature group definitions:
```yaml
feature_groups:
  person_features: ["person_cosine"]        # 98th percentile
  title_features: ["person_title_squared"]  # 95th percentile
  context_features: ["composite_cosine"]    # 90th percentile  
  binary_features: ["birth_death_match"]    # No scaling
```

## Production Features

### Checkpointing and Resumption
- **Persistent state**: Each stage saves checkpoints for resumption
- **Resume capability**: `python main.py --resume`
- **Status monitoring**: `python main.py --status`
- **Selective reset**: `python main.py --reset training classifying`

### Error Resilience
- Comprehensive exception handling with fallback strategies
- Retry logic with exponential backoff (Tenacity)
- Memory management and garbage collection
- Thread-safe caching with versioning

### Monitoring and Telemetry
- Detailed performance metrics collection
- Memory usage monitoring (psutil)
- Progress tracking (tqdm) 
- Configurable logging levels
- Transaction ID tracking for debugging

## Performance Results

**Latest Test Results (2025-06-05)**:
- **Test Size**: 14,930 entity pairs
- **Precision**: 99.55% (9,955 TP, 45 FP)
- **Recall**: 82.48% (2,114 FN out of 12,069 true matches)
- **F1 Score**: 90.22%
- **Accuracy**: 85.54%
- **Specificity**: 98.43%

The system achieves high precision with controlled recall, suitable for library catalog entity resolution where false positives are costly.

### Detailed Reporting Output

The reporting system generates comprehensive analysis including:

**Test Results Analysis** (`detailed_test_results_{timestamp}.csv`):
- Raw and normalized feature values for each test pair (`raw_` and `norm_` prefixes)
- Prediction confidence scores and correctness classification
- TP/FP/TN/FN categorization for each prediction

**Diagnostics** (`problematic_indicators_{timestamp}.json`):
- Detection of binary indicator anomalies (e.g., identical strings with incorrect indicator values)
- Feature engineering diagnostics and cache analysis
- Systematic quality assurance for feature calculation

**Interactive Dashboard** (`configuration_dashboard.html`):
- Parameter correlation analysis for optimization
- Performance metric trends across configurations
- Visual exploration of feature importance and model behavior

## Directory Structure

```
entity_resolver/
├── main.py                                    # CLI entry point
├── extract-names-benchmark-2024-12-16-csv.xq # XQuery BIBFRAME data extraction
├── config.yml                                 # Main configuration
├── scaling_config.yml                         # Feature scaling configuration
├── docker-compose.yml                         # Weaviate Docker setup
├── requirements.txt                           # Python dependencies
├── src/                                       # Core pipeline modules
│   ├── orchestrating.py                      # Pipeline orchestration
│   ├── preprocessing.py                      # CSV processing and deduplication
│   ├── embedding_and_indexing.py             # OpenAI + Weaviate integration
│   ├── feature_engineering.py                # Feature calculation and caching
│   ├── training.py                           # Logistic regression training
│   ├── classifying.py                        # Entity matching and clustering
│   ├── reporting.py                          # Results and visualization
│   ├── scaling_bridge.py                     # Feature scaling interface
│   └── utils.py                              # Utilities and helpers
├── setfit/                                    # SetFit taxonomy classifier
│   ├── train_setfit_classifier.py            # SetFit model training
│   ├── predict_setfit_classifier.py          # SetFit classification
│   └── SETFIT_README.md                      # SetFit documentation
├── data/
│   ├── input/                                 # Raw datasets and taxonomies
│   │   ├── training_dataset_classified.csv           # Extracted + classified data
│   │   ├── training_dataset_classified_2025-06-17.csv # Updated classifications
│   │   └── revised_taxonomy_final.json               # SetFit taxonomy structure
│   ├── checkpoints/                           # Pipeline state persistence
│   ├── output/                                # Results, reports, visualizations
│   │   ├── detailed_test_results_*.csv               # Complete prediction analysis
│   │   ├── test_summary_*.json                       # Performance metrics
│   │   ├── problematic_indicators_*.json             # Binary feature diagnostics
│   │   ├── feature_importance_*.png                  # Model weight visualizations
│   │   ├── configuration_dashboard.html              # Interactive analysis dashboard
│   │   ├── configuration_results.jsonl              # Structured optimization logs
│   │   └── setfit_prediction_discrepancy_report.csv  # Classification changes analysis
│   └── ground_truth/                          # Labeled entity pairs
│       └── labeled_matches.csv               # Training pairs (left,right,match)
└── logs/                                      # Pipeline execution logs
```

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build the pipeline image
docker build -t entity-resolver .

# Start Weaviate
docker-compose up -d weaviate

# Run pipeline
docker run --rm --network host \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.yml:/app/config.yml \
  --env-file .env \
  entity-resolver python main.py --config config.yml
```

### Option 2: Local Development

```bash
# Start Weaviate only
docker-compose up -d weaviate

# Run pipeline locally
source venv/bin/activate
python main.py --config config.yml
```

## Troubleshooting

### Common Issues

**Weaviate Connection**:
```bash
# Check Weaviate status
docker-compose ps
docker-compose logs weaviate
curl http://localhost:8080/v1/.well-known/ready
```

**OpenAI API Issues**:
- Verify API key: `echo $OPENAI_API_KEY`
- Check rate limits in OpenAI dashboard
- Monitor logs for API-related errors

**Memory Issues**:
- Reduce batch sizes in `config.yml`
- Increase Docker memory limits
- Monitor with `python main.py --status`

**Pipeline Failures**:
- Check logs in `data/logs/pipeline.log`
- Use `--start` and `--end` for problematic stages
- Enable debug logging: Set `log_level: DEBUG` in config

### Performance Optimization

**Batch Processing**:
- Adjust `embedding_batch_size` (default: 32)
- Tune `classification_batch_size` (default: 500)
- Balance workers vs batch size for your hardware

**Weaviate Tuning**:
```yaml
# In docker-compose.yml
VECTOR_INDEX_EF: 128           # Higher = better search quality
VECTOR_INDEX_EFCONSTRUCTION: 128  # Higher = better index quality  
VECTOR_INDEX_MAXCONNECTIONS: 64   # Higher = more connected graph
```

**Resource Allocation**:
```yaml
# In config.yml
embedding_workers: 4           # Parallel embedding generation
classification_workers: 8     # Parallel classification  
feature_workers: 4            # Parallel feature calculation
```

## Development

### Adding Custom Features

1. **Register feature function**:
   ```python
   # In src/custom_features.py
   def register_custom_features(feature_engineering, config):
       feature_engineering.register_feature('my_feature', my_feature_func)
   ```

2. **Enable in configuration**:
   ```yaml
   # In config.yml
   features:
     enabled: [..., "my_feature"]
   ```

### Running Tests

```bash
# Unit tests (if available)
pytest tests/

# Integration testing with small dataset
python main.py --config test_config.yml
```

### Extending the Pipeline

**New Pipeline Stage**:
1. Add stage function in `src/orchestrating.py`
2. Update `self.stages` list
3. Implement reset logic in `_reset_stage()`

**Alternative Scaling Strategy**:
1. Implement new scaler in `src/robust_scaler.py`
2. Add to `ScalingStrategy` enum in `src/scaling_bridge.py`
3. Configure in `scaling_config.yml`

## API Reference

### Key Classes

- `PipelineOrchestrator`: Main pipeline coordination
- `FeatureEngineering`: Feature calculation and caching
- `EntityClassifier`: Logistic regression training
- `EntityClassification`: Entity matching and clustering
- `ScalingBridge`: Feature scaling coordination

### Configuration Options

See `config.yml` for complete parameter documentation with defaults and descriptions.

## License

[License information to be added]

## Citation

If you use this entity resolution pipeline in your research, please cite:

```
[Citation information to be added]
```

---

**Contact**: [Contact information to be added]

**Repository**: [GitHub URL to be added]