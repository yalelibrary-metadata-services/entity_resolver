# Mistral Classifier Factory Tutorial: Replacing SetFit in Your Entity Resolution Pipeline

## Understanding the Opportunity

Your current entity resolution pipeline uses SetFit for taxonomy classification - the feature that determines whether a person is in "Music and Sound Arts" vs "Literature and Narrative Arts" vs other domains. Mistral's Classifier Factory offers a compelling alternative that could simplify your infrastructure while potentially improving performance.

## Why Consider Mistral Classifier Factory?

### Current SetFit Approach
- **Requires local GPU training** with sentence transformers
- **Memory intensive** (you experienced 29GB+ allocation issues)
- **Complex setup** with hierarchical classification logic
- **Self-managed infrastructure** for model serving

### Mistral Classifier Factory Benefits
- **Cloud-based training** (no local GPU needed)
- **Production-ready API** (no serving infrastructure)
- **Multi-target classification** (can classify multiple aspects simultaneously)
- **Cost-effective** ($4 per training job + $2/month storage)
- **Faster deployment** (minutes vs hours/days)

## Understanding Your Use Case

Your taxonomy classification currently distinguishes between domains like:
```json
{
  "Humanities, Thought, and Interpretation": {
    "Literature and Narrative Arts": ["Poetry", "Fiction", "Drama"],
    "Music and Sound Arts": ["Composition", "Performance", "Audio Arts"],
    "Visual Arts": ["Painting", "Sculpture", "Photography"]
  }
}
```

With Mistral Classifier Factory, you can enhance this to classify multiple aspects simultaneously.

## Step-by-Step Implementation Guide

### Step 1: Prepare Your Training Data

Transform your existing training data from SetFit format to Mistral's JSON format:

**Current SetFit Data Format:**
```csv
composite,person,roles,title,setfit_prediction,is_parent_category
"Contributor: Bach, Johann Sebastian...",Bach\, Johann Sebastian,Contributor,The Well-Tempered Clavier,Music and Sound Arts,FALSE
```

**Mistral Classifier Factory Format:**
```json
{
  "text": "Contributor: Bach, Johann Sebastian, 1685-1750\nTitle: The Well-Tempered Clavier\nSubjects: Keyboard music; Fugues",
  "labels": {
    "domain": "Music and Sound Arts",
    "subcategory": "Composition",
    "confidence_level": "high",
    "historical_period": "baroque"
  }
}
```

### Step 2: Enhanced Multi-Target Classification

Unlike SetFit's single classification, Mistral allows you to classify multiple aspects:

```json
{
  "text": "Contributor: Virginia Woolf, 1882-1941\nTitle: To the Lighthouse\nSubjects: Modernist literature; Stream of consciousness",
  "labels": {
    "domain": "Literature and Narrative Arts",
    "subcategory": "Fiction", 
    "literary_movement": "modernism",
    "historical_period": "early_20th_century",
    "gender": "female_author"
  }
}
```

This gives you much richer classification for entity resolution features.

### Step 3: Data Conversion Script

Create a script to convert your existing training data:

```python
import json
import pandas as pd
from pathlib import Path

def convert_setfit_to_mistral(setfit_csv_path: str, output_jsonl_path: str):
    """Convert SetFit training data to Mistral Classifier Factory format."""
    
    # Load your existing training data
    df = pd.read_csv(setfit_csv_path)
    
    # Define your enhanced taxonomy mapping
    enhanced_taxonomy = {
        "Music and Sound Arts": {
            "subcategories": ["Composition", "Performance", "Audio Arts", "Music Theory"],
            "typical_subjects": ["keyboard music", "orchestral", "vocal music", "jazz"]
        },
        "Literature and Narrative Arts": {
            "subcategories": ["Poetry", "Fiction", "Drama", "Literary Criticism"],
            "typical_subjects": ["novels", "poetry", "plays", "literary theory"]
        },
        # Add other domains...
    }
    
    training_examples = []
    
    for _, row in df.iterrows():
        # Create enhanced text representation
        text_parts = []
        
        if pd.notna(row['person']):
            text_parts.append(f"Person: {row['person']}")
        if pd.notna(row['title']):
            text_parts.append(f"Title: {row['title']}")
        if pd.notna(row['roles']):
            text_parts.append(f"Role: {row['roles']}")
        if pd.notna(row['subjects']):
            text_parts.append(f"Subjects: {row['subjects']}")
        
        text = "\n".join(text_parts)
        
        # Create enhanced labels
        domain = row['setfit_prediction']
        labels = {
            "domain": domain,
            "is_parent_category": "yes" if row['is_parent_category'] else "no"
        }
        
        # Add subcategory inference based on subjects/titles
        if domain in enhanced_taxonomy:
            subcategory = infer_subcategory(row, enhanced_taxonomy[domain])
            if subcategory:
                labels["subcategory"] = subcategory
        
        # Add temporal classification based on person name
        time_period = extract_time_period(row['person'])
        if time_period:
            labels["historical_period"] = time_period
            
        training_examples.append({
            "text": text,
            "labels": labels
        })
    
    # Write JSONL file
    with open(output_jsonl_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Converted {len(training_examples)} examples to {output_jsonl_path}")

def infer_subcategory(row, taxonomy_info):
    """Infer subcategory from subjects and titles."""
    subjects = str(row.get('subjects', '')).lower()
    title = str(row.get('title', '')).lower()
    
    for subcategory in taxonomy_info['subcategories']:
        for typical_subject in taxonomy_info['typical_subjects']:
            if typical_subject in subjects or typical_subject in title:
                return subcategory
    return None

def extract_time_period(person_name):
    """Extract historical period from person name with dates."""
    import re
    
    # Look for birth/death years in person name
    date_pattern = r'(\d{4})-(\d{4})'
    match = re.search(date_pattern, str(person_name))
    
    if match:
        birth_year = int(match.group(1))
        
        if birth_year < 1600:
            return "renaissance_or_earlier"
        elif birth_year < 1750:
            return "baroque"
        elif birth_year < 1820:
            return "classical"
        elif birth_year < 1900:
            return "romantic"
        elif birth_year < 1950:
            return "early_modern"
        else:
            return "contemporary"
    
    return None

# Usage
convert_setfit_to_mistral(
    "data/input/training_dataset_classified.csv",
    "data/output/mistral_training_data.jsonl"
)
```

### Step 4: Upload Data to Mistral

```python
import requests
import os

# Set your Mistral API key
MISTRAL_API_KEY = "your_mistral_api_key_here"

def upload_training_data(file_path: str) -> str:
    """Upload training data to Mistral and return file ID."""
    
    url = "https://api.mistral.ai/v1/files"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    
    with open(file_path, 'rb') as f:
        files = {
            'file': f,
            'purpose': ('', 'fine-tune')
        }
        
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        
        file_info = response.json()
        print(f"Uploaded {file_path}, file ID: {file_info['id']}")
        return file_info['id']

# Upload training and validation data
training_file_id = upload_training_data("data/output/mistral_training_data.jsonl")
validation_file_id = upload_training_data("data/output/mistral_validation_data.jsonl")
```

### Step 5: Create Fine-Tuning Job

```python
def create_classification_job(training_file_id: str, validation_file_id: str):
    """Create a Mistral Classifier Factory fine-tuning job."""
    
    url = "https://api.mistral.ai/v1/fine_tuning/jobs"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    job_config = {
        "model": "ministral-3b-latest",
        "job_type": "classifier", 
        "training_files": [{"file_id": training_file_id, "weight": 1}],
        "validation_files": [validation_file_id],
        "hyperparameters": {
            "training_steps": 100,  # Adjust based on your data size
            "learning_rate": 0.0001
        },
        "auto_start": True,  # Start training immediately
        "integrations": {
            "wandb": {
                "project": "yale-entity-resolution",
                "name": "mistral-taxonomy-classifier"
            }
        }
    }
    
    response = requests.post(url, headers=headers, json=job_config)
    response.raise_for_status()
    
    job_info = response.json()
    print(f"Created fine-tuning job: {job_info['id']}")
    print(f"Status: {job_info['status']}")
    
    return job_info['id']

job_id = create_classification_job(training_file_id, validation_file_id)
```

### Step 6: Monitor Training Progress

```python
def check_job_status(job_id: str):
    """Check the status of your fine-tuning job."""
    
    url = f"https://api.mistral.ai/v1/fine_tuning/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    job_info = response.json()
    
    print(f"Job Status: {job_info['status']}")
    print(f"Progress: {job_info.get('training_progress', 'N/A')}")
    
    if job_info['status'] == 'completed':
        print(f"Fine-tuned model: {job_info['fine_tuned_model']}")
        return job_info['fine_tuned_model']
    elif job_info['status'] == 'failed':
        print(f"Job failed: {job_info.get('error', 'Unknown error')}")
    
    return None

# Check status periodically
import time

model_id = None
while model_id is None:
    model_id = check_job_status(job_id)
    if model_id is None:
        print("Training in progress... checking again in 5 minutes")
        time.sleep(300)  # Wait 5 minutes
```

### Step 7: Integrate with Your Entity Resolution Pipeline

Replace your SetFit taxonomy classification with Mistral API calls:

```python
def get_mistral_classification(text: str, model_id: str) -> dict:
    """Get classification from your fine-tuned Mistral model."""
    
    url = "https://api.mistral.ai/v1/classifications"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "input": [text]
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return result['predictions'][0]

# Updated taxonomy dissimilarity calculation
def calc_mistral_taxonomy_dissimilarity(left_id: str, right_id: str, model_id: str) -> float:
    """Calculate taxonomy dissimilarity using Mistral classifier."""
    
    # Get composite text for both entities
    left_text = get_composite_text(left_id)
    right_text = get_composite_text(right_id)
    
    # Get classifications
    left_classification = get_mistral_classification(left_text, model_id)
    right_classification = get_mistral_classification(right_text, model_id)
    
    # Calculate dissimilarity based on multiple targets
    dissimilarity = 0.0
    weight_sum = 0.0
    
    # Domain dissimilarity (highest weight)
    if left_classification['domain'] != right_classification['domain']:
        dissimilarity += 1.0 * 0.5
    weight_sum += 0.5
    
    # Subcategory dissimilarity (medium weight)
    if 'subcategory' in left_classification and 'subcategory' in right_classification:
        if left_classification['subcategory'] != right_classification['subcategory']:
            dissimilarity += 1.0 * 0.3
        weight_sum += 0.3
    
    # Historical period dissimilarity (lower weight)
    if 'historical_period' in left_classification and 'historical_period' in right_classification:
        if left_classification['historical_period'] != right_classification['historical_period']:
            dissimilarity += 1.0 * 0.2
        weight_sum += 0.2
    
    return dissimilarity / weight_sum if weight_sum > 0 else 0.0
```

### Step 8: Performance Optimization

Implement caching to reduce API calls and costs:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_mistral_classification(text_hash: str, model_id: str) -> str:
    """Cached classification to avoid repeated API calls."""
    # This is a placeholder - implement actual caching logic
    # In production, use Redis or database caching
    pass

def get_cached_classification(text: str, model_id: str) -> dict:
    """Get classification with caching."""
    
    # Create hash of input text for cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first (implement your caching strategy)
    cached_result = get_from_cache(text_hash)
    if cached_result:
        return cached_result
    
    # Get fresh classification
    result = get_mistral_classification(text, model_id)
    
    # Cache result
    save_to_cache(text_hash, result)
    
    return result
```

## Cost Analysis and Comparison

### Mistral Classifier Factory Costs
- **Training**: $4 per job (one-time)
- **Storage**: $2/month per model
- **Inference**: ~$0.002 per 1K tokens (estimate)
- **Total for your pipeline**: ~$50-100/month including inference

### SetFit Costs (Current)
- **Training**: GPU time (hours on AWS/local)
- **Storage**: Model files (minimal)
- **Inference**: Local GPU/CPU compute
- **Total**: Variable, but includes infrastructure costs

### Benefits Beyond Cost
1. **Reduced complexity**: No model serving infrastructure
2. **Better scalability**: Handles traffic spikes automatically
3. **Multi-target classification**: Richer features for entity resolution
4. **Faster iteration**: Minutes to retrain vs hours
5. **Production reliability**: Mistral's uptime vs self-managed

## Integration Strategy

### Phase 1: Parallel Testing
Run both SetFit and Mistral classifiers in parallel to compare:
- Classification accuracy
- API latency vs local inference
- Cost per classification
- Feature quality for entity resolution

### Phase 2: A/B Testing
Use Mistral classifications for a subset of your entity resolution pipeline to measure:
- Overall precision/recall impact
- False positive/negative patterns
- Feature importance changes

### Phase 3: Full Migration
Replace SetFit with Mistral in production, monitoring:
- Pipeline performance metrics
- Cost efficiency
- Operational simplicity

## Advanced Applications

### Multi-Aspect Entity Resolution
With Mistral's multi-target classification, enhance your entity matching:

```python
def enhanced_entity_features(left_id: str, right_id: str, model_id: str):
    """Calculate enhanced features using multi-target classification."""
    
    left_class = get_cached_classification(get_composite_text(left_id), model_id)
    right_class = get_cached_classification(get_composite_text(right_id), model_id)
    
    features = {}
    
    # Domain dissimilarity (your current feature)
    features['taxonomy_dissimilarity'] = calc_taxonomy_dissimilarity(left_class, right_class)
    
    # New temporal consistency feature
    features['temporal_consistency'] = calc_temporal_consistency(left_class, right_class)
    
    # New genre/style consistency feature  
    features['style_consistency'] = calc_style_consistency(left_class, right_class)
    
    # Confidence-weighted feature
    features['classification_confidence'] = min(
        left_class.get('confidence', 1.0),
        right_class.get('confidence', 1.0)
    )
    
    return features
```

This tutorial shows how Mistral Classifier Factory can replace SetFit in your entity resolution pipeline while providing enhanced capabilities, reduced operational complexity, and potentially better performance. The key advantage is moving from a complex, self-managed ML system to a simple API integration that gives you more sophisticated classification capabilities.