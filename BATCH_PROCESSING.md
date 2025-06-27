# OpenAI Batch Processing for Entity Resolution

This document explains how to use the OpenAI Batch API for embedding generation in the entity resolution pipeline, which provides **50% cost savings** compared to real-time processing with a 24-hour turnaround time.

## Overview

The entity resolution pipeline now supports two modes for embedding generation:

1. **Real-time processing** (default) - Immediate processing with higher cost
2. **Batch processing** - Asynchronous processing with 50% cost savings and up to 24-hour turnaround

## Configuration

### Enabling Batch Processing

In your `config.yml`, set the following parameters:

```yaml
# Batch Processing Configuration
use_batch_embeddings: true  # Set to true to use OpenAI Batch API
batch_embedding_size: 50000  # Number of requests per batch file (max 50,000)
max_requests_per_file: 50000  # Maximum requests per JSONL file
batch_manual_polling: true  # Set to true for manual polling (recommended)
batch_poll_interval: 300  # Seconds between status polls when auto-polling
batch_max_wait_time: 86400  # Maximum wait time for batch completion (24 hours)
```

### Polling Modes

The system supports two polling modes:

1. **Manual Polling** (recommended, `batch_manual_polling: true`)

   - Creates batch jobs and exits immediately
   - You manually check status when convenient
   - No need to keep scripts running
2. **Automatic Polling** (`batch_manual_polling: false`)

   - Creates jobs and continuously polls for completion
   - Requires keeping the script running for up to 24 hours

### OpenAI API Key

Ensure your OpenAI API key is set in your environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to your `config.yml`:

```yaml
openai_api_key: "your-api-key-here"
```

## Usage

### Manual Polling Workflow (Recommended)

**Step 1: Create Batch Jobs**

```bash
# Enable batch processing in config.yml: use_batch_embeddings: true, batch_manual_polling: true
python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing
```

This will:

- Create JSONL files with embedding requests
- Upload files to OpenAI
- Create batch jobs
- Exit immediately with job IDs

**Step 2: Check Job Status (when convenient)**

```bash
# Check status anytime - no continuous polling required
python main.py --config config.yml --batch-status
```

Or use the dedicated batch manager:

```bash
python batch_manager.py --status
```

**Step 3: Download Results (when jobs complete)**

```bash
# Download and process completed results
python main.py --config config.yml --batch-results
```

Or use the batch manager:

```bash
python batch_manager.py --download
```

### Alternative: Using the Batch Manager

The batch manager script provides the same functionality as standalone commands:

```bash
# Create jobs
python batch_manager.py --create

# Check status (run anytime)
python batch_manager.py --status

# Download results when ready
python batch_manager.py --download
```

### Automatic Polling Workflow

1. If you prefer automatic polling (requires keeping script running):

```bash
# Set batch_manual_polling: false in config.yml
python main.py --config config.yml --start embedding_and_indexing --end embedding_and_indexing
```

The system will automatically handle all steps and wait for completion.

### Monitoring Progress

The batch processing includes detailed logging and progress tracking:

- **Job Creation**: Shows how many batch jobs are created
- **Status Polling**: Regular updates on job status with progress bars
- **Completion**: Reports on successful vs failed jobs
- **Cost Savings**: Estimates actual cost savings achieved

### Checkpointing and Recovery

Batch processing includes robust checkpointing:

- **Job Tracking**: All batch jobs are tracked in `batch_jobs.pkl`
- **Processed Hashes**: Completed items tracked in `batch_processed_hashes.pkl`
- **File Management**: JSONL request/result files are preserved for debugging
- **Resume Capability**: Can resume interrupted batch processing

## Workflow Details

### 1. Request File Creation

- Generates JSONL files with up to 50,000 embedding requests each
- Each request includes a unique `custom_id` for tracking
- Files are saved in the checkpoint directory

**Example JSONL Request Format:**

```jsonl
{"custom_id": "a1b2c3d4_composite_0", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Contributor: Smith, John\nTitle: Advanced Machine Learning Techniques\nRoles: Author, Editor", "encoding_format": "float"}}
{"custom_id": "e5f6g7h8_person_1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Smith, John", "encoding_format": "float"}}
{"custom_id": "i9j0k1l2_title_2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": "Advanced Machine Learning Techniques", "encoding_format": "float"}}
```

**Custom ID Format:** `{hash_value}_{field_type}_{index}`

- `hash_value`: CRC32 hash of the original string
- `field_type`: Field name (composite, person, title, etc.)
- `index`: Sequential index for uniqueness

### 2. Batch Job Management

- Uploads JSONL files to OpenAI
- Creates batch jobs with 24-hour completion window
- Tracks all jobs with metadata and status

### 3. Status Polling

- Polls job status every 5 minutes (configurable)
- Shows progress with descriptive status updates
- Handles all possible job states (pending, in_progress, completed, failed, etc.)

### 4. Result Processing

- Downloads completed job results
- Processes JSONL response files
- Extracts embeddings and prepares for indexing
- Handles partial failures gracefully

**Example JSONL Response Format:**

```jsonl
{"id": "batch_req_123", "custom_id": "a1b2c3d4_composite_0", "response": {"status_code": 200, "request_id": "req_456", "body": {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.123, -0.456, 0.789, ...]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 15, "total_tokens": 15}}}}
{"id": "batch_req_124", "custom_id": "e5f6g7h8_person_1", "response": {"status_code": 200, "request_id": "req_457", "body": {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [-0.321, 0.654, -0.987, ...]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 3, "total_tokens": 3}}}}
```

The system extracts the `embedding` arrays and maps them back to the original strings using the `custom_id`.

### 5. Weaviate Indexing

- Indexes embeddings in the same EntityString collection
- Uses identical UUID generation for consistency
- Maintains compatibility with real-time processing results

## Cost Analysis

### Example Cost Comparison

For 100,000 embedding requests using `text-embedding-3-small`:

- **Real-time**: ~$10.00 (standard pricing)
- **Batch**: ~$5.00 (50% discount)
- **Savings**: $5.00 (50% reduction)

The larger your dataset, the more significant the savings become.

## Error Handling

The batch processing includes comprehensive error handling:

- **Network Failures**: Automatic retries with exponential backoff
- **API Errors**: Detailed error logging and graceful degradation
- **Partial Failures**: Individual request failures don't stop the entire batch
- **Timeout Handling**: Configurable maximum wait times
- **File Corruption**: Validation and recovery mechanisms

## Comparison: Real-time vs Batch

| Feature               | Real-time                   | Batch                      |
| --------------------- | --------------------------- | -------------------------- |
| **Cost**        | Standard pricing            | 50% discount               |
| **Speed**       | Immediate                   | Up to 24 hours             |
| **Scalability** | Rate limited                | High throughput            |
| **Use Case**    | Development, small datasets | Production, large datasets |
| **Monitoring**  | Real-time progress          | Polling-based              |

## Best Practices

### When to Use Each Mode

✅ **Use manual polling for:**

- Production environments
- Large datasets
- When you don't want to keep scripts running
- Cost optimization priority

✅ **Use automatic polling for:**

- Development and testing
- Smaller datasets
- When immediate results are preferred
- Continuous integration pipelines

❌ **Avoid batch processing entirely for:**

- Very small datasets (<1,000 strings)
- Time-critical applications requiring immediate results
- Interactive development workflows

### Optimization Tips

1. **Batch Size**: Use the maximum batch size (50,000) for best cost efficiency
2. **Timing**: Start batch jobs during off-peak hours
3. **Monitoring**: Set up alerts for job completion
4. **Checkpointing**: Regularly save checkpoints for long-running processes

## Troubleshooting

### Common Issues

1. **Jobs Stuck in Pending**

   - Check OpenAI service status
   - Verify API key permissions
   - Monitor queue length
2. **High Failure Rate**

   - Check input data quality
   - Verify JSONL format
   - Review error messages in results
3. **Slow Processing**

   - Adjust poll interval for less frequent checks
   - Consider splitting very large datasets

### Reset and Recovery

To reset batch processing state:

```bash
python main.py --config config.yml --reset embedding_and_indexing
```

This will clean up all batch-related checkpoints and files.

### Manual Commands Reference

| Command                                | Description                            |
| -------------------------------------- | -------------------------------------- |
| `python batch_manager.py --create`   | Create batch jobs and upload to OpenAI |
| `python batch_manager.py --status`   | Check status of all batch jobs         |
| `python batch_manager.py --download` | Download and process completed results |
| `python main.py --batch-status`      | Check batch status via main script     |
| `python main.py --batch-results`     | Download results via main script       |

## Files Created

During batch processing, the following files are created in the checkpoint directory:

- `batch_requests_*.jsonl` - Request files uploaded to OpenAI
- `batch_results_*.jsonl` - Result files downloaded from OpenAI
- `batch_jobs.pkl` - Job tracking and metadata
- `batch_processed_hashes.pkl` - Processed item tracking

These files can be useful for debugging and audit purposes.
