# Batch API Integration with Weaviate Indexing

This document explains how the OpenAI Batch API integrates with Weaviate indexing, how field mappings are preserved, and how batch output files are processed.

## Overview

The batch processing system maintains full compatibility with the existing Weaviate indexing pipeline while adding asynchronous processing capabilities. Field mappings and data integrity are preserved through a sophisticated tracking system.

## Field Mapping Preservation Process

### 1. Initial Data Structure

The system starts with three key data structures from preprocessing:

```python
string_dict = {
    'a1b2c3d4': 'Smith, John',
    'e5f6g7h8': 'Advanced Machine Learning Techniques',
    'f9g0h1i2': 'Contributor: Smith, John\nTitle: Advanced Machine Learning\nRoles: Author',
    # ... more hash -> string mappings (each hash represents ONE unique string)
}

field_hash_mapping = {
    'a1b2c3d4': {'person': 15},        # "Smith, John" appears in person field 15 times
    'e5f6g7h8': {'title': 12},         # "Advanced Machine Learning" appears in title field 12 times  
    'f9g0h1i2': {'composite': 8},      # Full composite string appears in composite field 8 times
    # Each hash represents ONE unique string that appears in specific field types
}

string_counts = {
    'a1b2c3d4': 15,  # total occurrences of "Smith, John"
    'e5f6g7h8': 12,  # total occurrences of "Advanced Machine Learning"
    'f9g0h1i2': 8,   # total occurrences of the composite string
    # ... more counts
}
```

**Key Point**: Each hash represents a unique string. A person field string like "Smith, John" will never have the same hash as a composite field string because they contain different text.

### 2. String Selection for Embedding

The system selects strings based on which fields are configured for embedding:

```python
# Configuration specifies which fields to embed
embed_fields = ["composite", "person", "title"]

# Selection process finds strings that appear in these fields:
strings_to_process = [
    ('a1b2c3d4', 'Smith, John', 'person', 15),
    ('e5f6g7h8', 'Advanced Machine Learning Techniques', 'title', 12),
    ('f9g0h1i2', 'Contributor: Smith, John\nTitle: Advanced...', 'composite', 8),
    # Each tuple: (hash, string, field_type, frequency)
]
```

### 3. Custom ID Mapping Creation

When creating batch requests, the system generates a `custom_id_mapping` that preserves all original metadata:

```python
# From _create_batch_requests_file()
custom_id_mapping = {
    'a1b2c3d4_person_0': {
        'hash_value': 'a1b2c3d4',
        'original_string': 'Smith, John',
        'field_type': 'person',
        'frequency': 15,
        'index': 0
    },
    'e5f6g7h8_title_1': {
        'hash_value': 'e5f6g7h8', 
        'original_string': 'Advanced Machine Learning Techniques',
        'field_type': 'title',
        'frequency': 12,
        'index': 1
    },
    'f9g0h1i2_composite_2': {
        'hash_value': 'f9g0h1i2',
        'original_string': 'Contributor: Smith, John\nTitle: Advanced...',
        'field_type': 'composite',
        'frequency': 8,
        'index': 2
    },
    # ... one entry per string selected for embedding
}
```

**Key Features:**
- **Unique Custom IDs**: Format `{hash}_{field_type}_{index}` ensures uniqueness
- **Complete Metadata**: All original data preserved for reconstruction
- **Field-Specific Processing**: Each string processed for its specific field type
- **Frequency Preservation**: Usage counts maintained for indexing

### 4. JSONL Request Generation

Each entry in the mapping becomes a batch request:

```jsonl
{"custom_id": "a1b2c3d4_person_0", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Smith, John", "encoding_format": "float"}}
{"custom_id": "e5f6g7h8_title_1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Advanced Machine Learning Techniques", "encoding_format": "float"}}
{"custom_id": "f9g0h1i2_composite_2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Contributor: Smith, John\nTitle: Advanced Machine Learning\nRoles: Author", "encoding_format": "float"}}
```

## Batch Output File Processing

### 1. Response Structure

OpenAI returns results in JSONL format with the custom IDs preserved:

```jsonl
{"id": "batch_req_123", "custom_id": "a1b2c3d4_person_0", "response": {"status_code": 200, "request_id": "req_456", "body": {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.123, -0.456, 0.789, ...]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 3, "total_tokens": 3}}}}
{"id": "batch_req_124", "custom_id": "e5f6g7h8_title_1", "response": {"status_code": 200, "request_id": "req_457", "body": {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [-0.321, 0.654, -0.987, ...]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 5, "total_tokens": 5}}}}
```

### 2. Result Processing Pipeline

The `_process_batch_results()` method reconstructs the original data:

```python
def _process_batch_results(self, results_path: str, custom_id_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    items_to_index = []
    
    with open(results_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            
            # Retrieve original metadata using custom_id
            item_data = custom_id_mapping[custom_id]
            
            # Extract embedding from response
            embedding = result['response']['body']['data'][0]['embedding']
            
            # Reconstruct complete item for indexing
            items_to_index.append({
                'hash_value': item_data['hash_value'],      # Original hash
                'original_string': item_data['original_string'],  # Original text
                'field_type': item_data['field_type'],      # Field context
                'frequency': item_data['frequency'],        # Usage frequency
                'vector': np.array(embedding, dtype=np.float32)  # Generated embedding
            })
    
    return items_to_index
```

### 3. Weaviate Indexing

The `_index_embeddings_batch()` method creates Weaviate objects with identical structure to real-time processing:

```python
def _index_embeddings_batch(self, items_to_index: List[Dict[str, Any]]) -> int:
    with self.collection.batch.fixed_size(batch_size=100) as batch_writer:
        for item in items_to_index:
            # Generate deterministic UUID (same as real-time processing)
            uuid_input = f"{item['hash_value']}_{item['field_type']}"
            uuid = generate_uuid5(uuid_input)
            
            # Create Weaviate object
            batch_writer.add_object(
                properties={
                    'original_string': item['original_string'],
                    'hash_value': item['hash_value'],
                    'field_type': item['field_type'],
                    'frequency': item['frequency']
                },
                uuid=uuid,
                vector=item['vector'].tolist()
            )
```

**Key Points:**
- **Identical UUIDs**: Uses same UUID generation as real-time processing (`hash_value_field_type`)
- **Same Schema**: Creates objects with identical properties structure
- **Field Separation**: Each field type gets appropriate embeddings for its string content
- **Deterministic**: Re-running batch processing produces identical results

## Command Reference

### Manual Processing Commands

**Create Batch Jobs:**
```bash
# Using main pipeline
python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing

# Using batch manager
python batch_manager.py --create
```

**Check Job Status:**
```bash
# Using main pipeline
python main.py --config config.yml --batch-status

# Using batch manager  
python batch_manager.py --status
```

**Retrieve, Process, and Index Results:**
```bash
# Using main pipeline (recommended)
python main.py --config config.yml --batch-results

# Using batch manager
python batch_manager.py --download
```

The `--batch-results` command performs three operations:
1. **Retrieves** completed batch results from OpenAI
2. **Processes** JSONL files to extract embeddings and reconstruct metadata
3. **Indexes** results in Weaviate with proper field mappings

## Data Flow Diagram

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Preprocessing     │    │    Batch Creation    │    │   OpenAI Batch API  │
│                     │    │                      │    │                     │
│ string_dict         │───▶│ custom_id_mapping    │───▶│ JSONL requests      │
│ field_hash_mapping  │    │ batch_requests.jsonl │    │ Job processing      │
│ string_counts       │    │                      │    │ batch_results.jsonl │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                  │
┌─────────────────────┐    ┌──────────────────────┐              │
│     Weaviate        │◄───│   Result Processing  │◄─────────────┘
│                     │    │                      │
│ EntityString        │    │ Extract embeddings   │
│ Collection          │    │ Reconstruct metadata │
│ - hash_value        │    │ Prepare for indexing │
│ - original_string   │    │                      │
│ - field_type        │    │                      │
│ - frequency         │    │                      │
│ - vector            │    │                      │
└─────────────────────┘    └──────────────────────┘
```

## Field Type Handling

### Why Field Types Matter

The system processes different strings for different field contexts:

```python
# Different field types contain different content:
"Smith, John" → person field
"Advanced Machine Learning" → title field  
"Contributor: Smith, John\nTitle: Advanced Machine Learning\nRoles: Author" → composite field

# Results in separate Weaviate objects with different UUIDs:
# UUID: hash_person → field_type: "person", content: "Smith, John"
# UUID: hash_title → field_type: "title", content: "Advanced Machine Learning"
# UUID: hash_composite → field_type: "composite", content: full composite string
```

This allows field-specific similarity searches in the entity resolution pipeline:

- **Person searches**: Find similar person names
- **Composite searches**: Find similar composite descriptions
- **Title searches**: Find similar work titles

### Consistency with Real-time Processing

Both batch and real-time processing create identical Weaviate structures:

| Aspect | Real-time | Batch | 
|--------|-----------|-------|
| **UUID Generation** | `generate_uuid5(f"{hash}_{field}")` | `generate_uuid5(f"{hash}_{field}")` |
| **Object Properties** | Same 4 fields | Same 4 fields |
| **Vector Dimensions** | 1536 (configurable) | 1536 (configurable) |
| **Field Separation** | Per field type | Per field type |
| **Hash Values** | CRC32 | CRC32 |

## Error Handling

### Partial Failures

The system handles partial failures gracefully:

```python
# If some embeddings fail
successful_requests = 45000
failed_requests = 5000

# Only successful embeddings are indexed
# Failed requests are logged but don't stop processing
# Can be retried by re-running the command
```

### Resume Capability

```python
# Tracks processed hashes to avoid duplicates
processed_hashes = {'a1b2c3d4', 'e5f6g7h8', 'f9g0h1i2', ...}

# Skip already processed items
if hash_val in self.processed_hashes:
    continue  # Skip this embedding
```

This allows safe re-running of batch processing commands without creating duplicates in Weaviate.

## Example: Complete Processing Flow

### Input Data
```python
# From preprocessing
string_dict = {
    'abc123': 'Tolkien, J.R.R.',
    'def456': 'The Lord of the Rings', 
    'ghi789': 'Contributor: Tolkien, J.R.R.\nTitle: The Lord of the Rings\nRoles: Author'
}

field_hash_mapping = {
    'abc123': {'person': 25},     # "Tolkien, J.R.R." in person field
    'def456': {'title': 18},      # "The Lord of the Rings" in title field
    'ghi789': {'composite': 12}   # Full string in composite field
}
```

### Batch Requests Created
```jsonl
{"custom_id": "abc123_person_0", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Tolkien, J.R.R.", "encoding_format": "float"}}
{"custom_id": "def456_title_1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "The Lord of the Rings", "encoding_format": "float"}}
{"custom_id": "ghi789_composite_2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Contributor: Tolkien, J.R.R.\nTitle: The Lord of the Rings\nRoles: Author", "encoding_format": "float"}}
```

### Weaviate Objects Created
```python
# Three separate objects in EntityString collection:
{
    "uuid": "generated_from_abc123_person",
    "properties": {
        "hash_value": "abc123",
        "original_string": "Tolkien, J.R.R.",
        "field_type": "person", 
        "frequency": 25
    },
    "vector": [0.1, -0.2, 0.3, ...]  # Embedding for "Tolkien, J.R.R."
}

{
    "uuid": "generated_from_def456_title", 
    "properties": {
        "hash_value": "def456",
        "original_string": "The Lord of the Rings",
        "field_type": "title",
        "frequency": 18
    },
    "vector": [0.4, -0.5, 0.6, ...]  # Embedding for "The Lord of the Rings"
}

{
    "uuid": "generated_from_ghi789_composite",
    "properties": {
        "hash_value": "ghi789", 
        "original_string": "Contributor: Tolkien, J.R.R.\nTitle: The Lord of the Rings\nRoles: Author",
        "field_type": "composite",
        "frequency": 12
    },
    "vector": [0.7, -0.8, 0.9, ...]  # Embedding for full composite string
}
```

This structure enables field-specific similarity searches while maintaining complete data integrity and compatibility with the existing entity resolution pipeline.