#!/usr/bin/env python3
"""
Parallel Individual Record Classification Script

Optimized for high-tier Anthropic accounts with high rate limits.
Uses asyncio and aiohttp for maximum throughput.
"""

import json
import os
import pandas as pd
import asyncio
import aiohttp
import time
import argparse
import logging
from tqdm.asyncio import tqdm
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("individual_classification_verification_parallel.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parallel Individual Record Classification using Anthropic API')
    parser.add_argument('--csv', required=True, help='Path to the training dataset CSV')
    parser.add_argument('--taxonomy', required=True, help='Path to the taxonomy JSON-LD')
    parser.add_argument('--output', required=True, help='Path for the output classification results JSON')
    parser.add_argument('--batch-size', type=int, default=200, help='Number of records to process in a batch')
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent API requests (optimized for 200K token/min limit)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--api-key', help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    parser.add_argument('--record-filter', help='Only process specific record index (e.g., "42")')
    parser.add_argument('--dry-run', action='store_true', help='Run without making API calls or writing files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging of prompts')
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this record index')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process')
    parser.add_argument('--rate-limit-rpm', type=int, default=4000, help='Rate limit in requests per minute')
    parser.add_argument('--save-interval', type=int, default=100, help='Save progress every N completed records')
    return parser.parse_args()

@dataclass
class ClassificationTask:
    """Represents a single classification task."""
    record_id: int
    person_id: str
    record_data: Dict[str, Any]

class AsyncRateLimiter:
    """Async rate limiter optimized for high throughput."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        self.tokens = min(100, requests_per_minute)  # Start with some tokens
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token with minimal waiting."""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Replenish tokens based on time passed
            new_tokens = time_passed * self.requests_per_second
            self.tokens = min(self.requests_per_minute / 60 * 2, self.tokens + new_tokens)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Brief wait if no tokens available
            wait_time = max(0.01, (1 - self.tokens) / self.requests_per_second)
            await asyncio.sleep(wait_time)
            self.tokens = max(0, self.tokens - 1)

def load_data(args, logger):
    """Load all required data files."""
    logger.info("Loading CSV data from %s", args.csv)
    df = pd.read_csv(args.csv)
    
    # Add a record index for tracking
    df.reset_index(drop=True, inplace=True)
    df['record_id'] = df.index
    
    logger.info(f"CSV contains {len(df)} records")
    
    logger.info("Loading taxonomy from %s", args.taxonomy)
    with open(args.taxonomy, 'r') as f:
        taxonomy = json.load(f)
    
    # Extract taxonomy information in a more usable format
    taxonomy_concepts = extract_taxonomy_concepts(taxonomy)
    
    return df, taxonomy_concepts

def extract_taxonomy_concepts(taxonomy):
    """Extract concepts from taxonomy in a more usable format."""
    taxonomy_concepts = {}
    for top_concept in taxonomy.get("skos:hasTopConcept", []):
        concept_id = top_concept.get("@id", "").split("/")[-1]
        pref_label = top_concept.get("skos:prefLabel", {}).get("@value", "")
        definition = top_concept.get("skos:definition", {}).get("@value", "")
        scope_note = top_concept.get("skos:scopeNote", {}).get("@value", "")
        
        # Extract alt labels for top concept
        alt_labels = []
        for alt_label in top_concept.get("skos:altLabel", []):
            alt_labels.append(alt_label.get("@value", ""))
        
        taxonomy_concepts[concept_id] = {
            "label": pref_label,
            "definition": definition,
            "scope_note": scope_note,
            "alt_labels": alt_labels,
            "narrower": {}
        }
        
        for narrower in top_concept.get("skos:narrower", []):
            narrower_id = narrower.get("@id", "").split("/")[-1]
            narrower_label = narrower.get("skos:prefLabel", {}).get("@value", "")
            narrower_definition = narrower.get("skos:definition", {}).get("@value", "")
            narrower_scope_note = narrower.get("skos:scopeNote", {}).get("@value", "")
            
            # Extract alt labels for narrower concept
            narrower_alt_labels = []
            for alt_label in narrower.get("skos:altLabel", []):
                narrower_alt_labels.append(alt_label.get("@value", ""))
            
            taxonomy_concepts[concept_id]["narrower"][narrower_id] = {
                "label": narrower_label,
                "definition": narrower_definition,
                "scope_note": narrower_scope_note,
                "alt_labels": narrower_alt_labels
            }
    
    return taxonomy_concepts

def extract_json_from_response(response_text, logger):
    """Extract JSON object from response text."""
    logger.debug("Attempting to extract JSON from response")
    
    # Try to extract JSON from code blocks first
    json_code_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
    if json_code_block_match:
        logger.debug("Found JSON code block")
        try:
            json_str = json_code_block_match.group(1)
            logger.debug(f"Extracted JSON string: {json_str[:200]}...")
            parsed = json.loads(json_str)
            
            # Validate required fields
            if (isinstance(parsed, dict) and 'label' in parsed and 'path' in parsed and 'rationale' in parsed):
                logger.debug("JSON has required fields")
                # Convert strings to arrays if needed
                if isinstance(parsed['label'], str):
                    parsed['label'] = [parsed['label']]
                if isinstance(parsed['path'], str):
                    parsed['path'] = [parsed['path']]
                
                if isinstance(parsed['label'], list) and isinstance(parsed['path'], list):
                    logger.debug("JSON validation successful")
                    return parsed
                else:
                    logger.debug("JSON validation failed - wrong types for label/path")
            else:
                logger.debug("JSON missing required fields")
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error in code block: {e}")
    else:
        logger.debug("No JSON code block found")
    
    # Try to find JSON object in the text
    try:
        json_pattern = r'({[\s\S]*?})'
        matches = re.finditer(json_pattern, response_text)
        
        best_match = None
        best_match_len = 0
        
        for match in matches:
            match_text = match.group(1)
            if len(match_text) > best_match_len:
                try:
                    parsed = json.loads(match_text)
                    if (isinstance(parsed, dict) and 'label' in parsed and 'path' in parsed and 'rationale' in parsed):
                        # Convert strings to arrays if needed
                        if isinstance(parsed['label'], str):
                            parsed['label'] = [parsed['label']]
                        if isinstance(parsed['path'], str):
                            parsed['path'] = [parsed['path']]
                            
                        if isinstance(parsed['label'], list) and isinstance(parsed['path'], list):
                            best_match = parsed
                            best_match_len = len(match_text)
                except json.JSONDecodeError:
                    continue
        
        if best_match:
            return best_match
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
    
    logger.error("Could not extract valid JSON from response")
    return None

def estimate_tokens(text: str) -> int:
    """Rough token estimation (approximately 4 characters per token for English)."""
    return len(text) // 4

def prepare_taxonomy_prompt(taxonomy_concepts):
    """Prepare a concise prompt with taxonomy information."""
    prompt = "Taxonomy Categories:\n\n"
    
    for concept_id, concept in taxonomy_concepts.items():
        prompt += f"# {concept['label']}\n"
        prompt += f"{concept['definition']}\n"
        
        if concept['alt_labels']:
            alt_labels_str = ", ".join(concept['alt_labels'])
            prompt += f"Associated terms: {alt_labels_str}\n"
        
        prompt += "\n"
        
        for narrower_id, narrower in concept['narrower'].items():
            prompt += f"## {narrower['label']}\n"
            prompt += f"{narrower['definition']}\n"
            
            if narrower['alt_labels']:
                narrower_alt_labels_str = ", ".join(narrower['alt_labels'])
                prompt += f"Associated terms: {narrower_alt_labels_str}\n"
            
            prompt += "\n"
    
    return prompt

# Global taxonomy prompt to avoid recreating it for each request
_taxonomy_prompt_cache = None

async def classify_record_async(session: aiohttp.ClientSession, task: ClassificationTask, 
                              taxonomy_prompt: str, args, logger, rate_limiter: AsyncRateLimiter):
    """Classify a single record asynchronously."""
    
    record_data = task.record_data
    
    # Prepare context
    context = f"Analyze this catalog entry for classification:\n\n"
    context += f"Person: {record_data.get('person', 'Unknown')}\n"
    context += f"PersonID: {record_data.get('personId', 'Unknown')}\n"
    context += f"Identity: {record_data.get('identity', 'Unknown')}\n"
    context += f"Composite: {record_data.get('composite', 'Unknown')}\n\n"
    
    # Include current classification if available
    if 'classification_label' in record_data and pd.notna(record_data['classification_label']):
        context += f"Current classification: {record_data['classification_label']}\n"
    if 'classification_path' in record_data and pd.notna(record_data['classification_path']):
        context += f"Current path: {record_data['classification_path']}\n"
    
    # Main prompt
    prompt = f"""Based on this catalog entry, determine the appropriate classification.

{taxonomy_prompt}

GUIDELINES:
- Use exact taxonomy concept names for labels
- May assign up to THREE concepts when evidence warrants it
- Primary classification should be most dominant domain
- Focus on THIS record's specific evidence

Respond in JSON format:
{{
  "label": ["Primary label", "Secondary (if applicable)", "Tertiary (if applicable)"],
  "path": ["Primary path", "Secondary path (if applicable)", "Tertiary path (if applicable)"],
  "rationale": "Detailed explanation with evidence from the catalog entry"
}}

Path format: "Parent Category > Subcategory"
"""

    # Wait for rate limit
    await rate_limiter.acquire()
    
    # API request
    api_url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": args.api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8000,
        "temperature": 0.0,
        "system": "You are an expert in library cataloging and classification who analyzes catalog entries to determine appropriate classification categories.",
        "messages": [{"role": "user", "content": context + prompt}]
    }
    
    try:
        logger.debug(f"→ Making API request for PersonID {task.person_id}")
        async with session.post(api_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            logger.debug(f"← Got response for PersonID {task.person_id}: status {response.status}")
            
            if response.status == 200:
                response_data = await response.json()
                result_text = response_data['content'][0]['text']
                logger.debug(f"Response text length for PersonID {task.person_id}: {len(result_text)} chars")
                
                result = extract_json_from_response(result_text, logger)
                if result:
                    logger.debug(f"Successfully parsed JSON for PersonID {task.person_id}")
                    return task.person_id, result
                else:
                    logger.error(f"Failed to parse JSON for PersonID {task.person_id}")
                    logger.error(f"Response text sample: {result_text[:500]}...")
                    return task.person_id, None
            else:
                error_text = await response.text()
                logger.error(f"API error for PersonID {task.person_id}: {response.status} - {error_text}")
                return task.person_id, None
                
    except asyncio.TimeoutError:
        logger.error(f"Timeout for PersonID {task.person_id}")
        return task.person_id, None
    except Exception as e:
        logger.error(f"Exception for PersonID {task.person_id}: {e}")
        return task.person_id, None

async def process_batch_parallel(tasks: List[ClassificationTask], taxonomy_prompt: str, 
                               args, logger, rate_limiter: AsyncRateLimiter):
    """Process a batch of tasks in parallel with retry logic."""
    
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def process_with_retry(task):
        async with semaphore:
            for attempt in range(args.max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        return await classify_record_async(session, task, taxonomy_prompt, args, logger, rate_limiter)
                except Exception as e:
                    if attempt < args.max_retries:
                        wait_time = min(2 ** attempt, 8)  # Cap exponential backoff
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All retries failed for PersonID {task.person_id}: {e}")
                        return task.person_id, None
    
    # Process all tasks in parallel
    logger.info(f"→ Launching {len(tasks)} parallel tasks...")
    results = await asyncio.gather(*[process_with_retry(task) for task in tasks], return_exceptions=True)
    logger.info(f"← All {len(tasks)} API requests completed")
    
    # Handle any exceptions
    final_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            final_results.append((None, None))
        else:
            final_results.append(result)
    
    return final_results

async def process_records_async(df: pd.DataFrame, taxonomy_concepts: Dict, args, logger):
    """Main async processing function."""
    results = {}
    
    # Apply filters
    start_idx = args.start_index
    end_idx = len(df)
    
    if args.max_records:
        end_idx = min(start_idx + args.max_records, len(df))
    
    if args.record_filter:
        try:
            record_idx = int(args.record_filter)
            if 0 <= record_idx < len(df):
                start_idx = record_idx
                end_idx = record_idx + 1
                logger.info(f"Processing only record index: {record_idx}")
            else:
                logger.error(f"Record filter {record_idx} out of range")
                return results
        except ValueError:
            logger.error(f"Invalid record filter: {args.record_filter}")
            return results
    
    records_to_process = df.iloc[start_idx:end_idx]
    logger.info(f"Processing {len(records_to_process)} records")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
        return results
    
    # Prepare taxonomy prompt once
    taxonomy_prompt = prepare_taxonomy_prompt(taxonomy_concepts)
    taxonomy_tokens = estimate_tokens(taxonomy_prompt)
    estimated_tokens_per_request = taxonomy_tokens + 500
    logger.info(f"Taxonomy prompt prepared ({len(taxonomy_prompt)} characters, ~{taxonomy_tokens} tokens)")
    logger.info(f"Estimated tokens per request: ~{estimated_tokens_per_request} (taxonomy + context + prompt)")
    
    # Calculate safe concurrency based on token limits (for informational purposes only)
    token_limit_per_minute = 200000  # Your input token limit
    max_safe_concurrency = max(1, token_limit_per_minute // (estimated_tokens_per_request * 60))
    projected_token_usage = args.concurrency * estimated_tokens_per_request * 60
    
    logger.info(f"Token usage analysis:")
    logger.info(f"  Recommended max concurrency: {max_safe_concurrency}")
    logger.info(f"  Your requested concurrency: {args.concurrency}")
    logger.info(f"  Projected token usage: {projected_token_usage:,} tokens/min")
    
    if projected_token_usage > token_limit_per_minute:
        logger.warning(f"⚠️  Projected usage ({projected_token_usage:,}) exceeds limit ({token_limit_per_minute:,})")
        logger.warning(f"⚠️  You may hit rate limits. Consider reducing --concurrency to {max_safe_concurrency}")
    else:
        logger.info(f"✓ Projected usage is within token limits")
    
    logger.info(f"Using concurrency: {args.concurrency}")
    
    # Create rate limiter
    rate_limiter = AsyncRateLimiter(args.rate_limit_rpm)
    
    # Create tasks
    tasks = []
    for idx, record in records_to_process.iterrows():
        record_data = record.to_dict()
        person_id = str(record_data.get('personId', 'Unknown'))
        
        task = ClassificationTask(
            record_id=idx,
            person_id=person_id,
            record_data=record_data
        )
        tasks.append(task)
    
    logger.info(f"Created {len(tasks)} classification tasks")
    logger.info(f"Using concurrency: {args.concurrency}, Rate limit: {args.rate_limit_rpm} RPM")
    
    # Process in batches
    start_time = time.time()
    completed_count = 0
    
    for i in range(0, len(tasks), args.batch_size):
        batch = tasks[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")
        logger.info(f"→ Starting {args.concurrency} concurrent API requests...")
        
        # Process batch
        batch_start = time.time()
        batch_results = await process_batch_parallel(batch, taxonomy_prompt, args, logger, rate_limiter)
        batch_time = time.time() - batch_start
        
        logger.info(f"← Batch processing completed in {batch_time:.1f}s")
        
        # Store results and log individual completions
        batch_success = 0
        for person_id, result in batch_results:
            if person_id and result:
                results[person_id] = {
                    'label': result['label'] if isinstance(result['label'], list) else [result['label']],
                    'path': result['path'] if isinstance(result['path'], list) else [result['path']],
                    'rationale': result['rationale']
                }
                
                # Log individual classification
                primary_label = result['label'][0] if isinstance(result['label'], list) else result['label']
                logger.info(f"✓ Classified PersonID {person_id}: {primary_label}")
                
                # Log multiple classifications if present
                if isinstance(result['label'], list) and len(result['label']) > 1:
                    all_labels = ", ".join(result['label'])
                    logger.info(f"  → Multiple classifications: {all_labels}")
                
                batch_success += 1
                completed_count += 1
                
                # Save progress every 5 completions for more frequent updates
                if completed_count % 5 == 0:
                    with open(f"{args.output}.partial", 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.debug(f"Saved progress: {len(results)} classifications")
            else:
                logger.warning(f"✗ Failed to classify PersonID {person_id}")
        
        logger.info(f"Batch {batch_num}/{total_batches} completed: {batch_success}/{len(batch)} successful in {batch_time:.1f}s")
        logger.info(f"Rate: {len(batch)/batch_time:.1f} records/sec | Total completed: {completed_count}/{len(tasks)}")
        
        # Save progress periodically
        if completed_count % args.save_interval == 0 or batch_num == total_batches:
            logger.info(f"Saving progress... ({completed_count} completed)")
            with open(f"{args.output}.partial", 'w') as f:
                json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"Processing completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Overall rate: {len(results)/total_time:.1f} records/sec")
    logger.info(f"Successfully classified {len(results)}/{len(tasks)} records")
    
    return results

async def main_async():
    """Main async function."""
    args = parse_arguments()
    logger = setup_logging()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        # Enable debug for API requests even in normal mode for better visibility
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    logger.info("Starting Parallel Individual Record Classification")
    logger.info(f"Configuration: {args.concurrency} concurrent, {args.rate_limit_rpm} RPM limit")
    logger.info("Note: Concurrency optimized for 200K input tokens/minute limit")
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("No Anthropic API key provided")
        return
    
    args.api_key = api_key
    
    # Load data
    logger.info("Loading data...")
    df, taxonomy_concepts = load_data(args, logger)
    
    logger.info(f"Loaded taxonomy with {len(taxonomy_concepts)} top-level concepts")
    
    # Process records
    results = await process_records_async(df, taxonomy_concepts, args, logger)
    
    # Write final output
    if not args.dry_run and results:
        logger.info("Writing final results...")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Clean up partial file
        if os.path.exists(f"{args.output}.partial"):
            os.remove(f"{args.output}.partial")
        
        logger.info(f"Results written to {args.output}")
        logger.info(f"Final summary: {len(results)} PersonIDs classified")
    else:
        logger.info("Dry run completed")

def main():
    """Main entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()