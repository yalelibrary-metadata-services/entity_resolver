import json
import os
import pandas as pd
from anthropic import Anthropic
import time
import argparse
import logging
from tqdm import tqdm
import re

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("individual_classification_verification.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify and correct individual record classifications using Anthropic API')
    parser.add_argument('--csv', required=True, help='Path to the training dataset CSV')
    parser.add_argument('--taxonomy', required=True, help='Path to the taxonomy JSON-LD')
    parser.add_argument('--output', required=True, help='Path for the output classification results JSON')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of records to process in a batch')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay in seconds between API calls')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API calls')
    parser.add_argument('--api-key', help='Anthropic API key (defaults to ANTHROPIC_API_KEY env var)')
    parser.add_argument('--record-filter', help='Only process specific record index (e.g., "42")')
    parser.add_argument('--dry-run', action='store_true', help='Run without making API calls or writing files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging of prompts')
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this record index')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process')
    return parser.parse_args()

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
    
    # Log a snippet of the response text for debugging
    preview_length = min(300, len(response_text))
    logger.debug(f"Response text preview: {response_text[:preview_length]}...")
    
    # Try to extract JSON from code blocks first
    json_code_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
    if json_code_block_match:
        try:
            json_str = json_code_block_match.group(1)
            logger.debug(f"Found JSON in code block: {json_str}")
            parsed = json.loads(json_str)
            
            # Validate that the JSON has the required fields and correct format
            if (isinstance(parsed, dict) and 
                'label' in parsed and 
                'path' in parsed and 
                'rationale' in parsed):
                
                # Convert string values to single-item arrays if needed
                if isinstance(parsed['label'], str):
                    logger.debug(f"Converting 'label' from string to array: {parsed['label']}")
                    parsed['label'] = [parsed['label']]
                if isinstance(parsed['path'], str):
                    logger.debug(f"Converting 'path' from string to array: {parsed['path']}")
                    parsed['path'] = [parsed['path']]
                
                # Ensure we have arrays for label and path
                if isinstance(parsed['label'], list) and isinstance(parsed['path'], list):
                    logger.debug(f"Successfully extracted JSON with {len(parsed['label'])} classifications: {parsed['label']}")
                    return parsed
                else:
                    logger.warning(f"Invalid format for 'label' or 'path': {type(parsed['label'])}, {type(parsed['path'])}")
            else:
                logger.warning("JSON from code block missing required fields or has incorrect format")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from code block: {e}")
    else:
        logger.debug("No JSON code block found, attempting to extract from plain text")
    
    # Try to find JSON object in the text
    try:
        # Look for the outermost JSON object
        json_pattern = r'({[\s\S]*?})'
        matches = re.finditer(json_pattern, response_text)
        
        # Find the longest match, which is likely the complete JSON
        best_match = None
        best_match_len = 0
        
        for match in matches:
            match_text = match.group(1)
            if len(match_text) > best_match_len:
                try:
                    # Verify it's valid JSON
                    parsed = json.loads(match_text)
                    # Check for required fields
                    if (isinstance(parsed, dict) and 
                        'label' in parsed and 
                        'path' in parsed and 
                        'rationale' in parsed):
                        
                        # Convert string values to single-item arrays if needed
                        if isinstance(parsed['label'], str):
                            logger.debug(f"Converting 'label' from string to array: {parsed['label']}")
                            parsed['label'] = [parsed['label']]
                        if isinstance(parsed['path'], str):
                            logger.debug(f"Converting 'path' from string to array: {parsed['path']}")
                            parsed['path'] = [parsed['path']]
                            
                        # Ensure we have arrays for label and path
                        if isinstance(parsed['label'], list) and isinstance(parsed['path'], list):
                            best_match = parsed
                            best_match_len = len(match_text)
                except json.JSONDecodeError:
                    continue
        
        if best_match:
            logger.debug(f"Successfully extracted JSON with {len(best_match['label'])} classifications: {best_match['label']}")
            return best_match
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
    
    logger.error("Could not extract valid JSON from response")
    # Log more of the response for error analysis
    logger.error(f"Failed response content (first 1000 chars): {response_text[:1000]}")
    return None

def prepare_taxonomy_prompt(taxonomy_concepts):
    """Prepare a concise prompt with taxonomy information."""
    prompt = "Taxonomy Categories:\n\n"
    
    for concept_id, concept in taxonomy_concepts.items():
        prompt += f"# {concept['label']}\n"
        prompt += f"{concept['definition']}\n"
        
        # Add alt labels for top concept
        if concept['alt_labels']:
            alt_labels_str = ", ".join(concept['alt_labels'])
            prompt += f"Associated terms or alternative labels: {alt_labels_str}\n"
        
        prompt += "\n"
        
        for narrower_id, narrower in concept['narrower'].items():
            prompt += f"## {narrower['label']}\n"
            prompt += f"{narrower['definition']}\n"
            
            # Add alt labels for narrower concept
            if narrower['alt_labels']:
                narrower_alt_labels_str = ", ".join(narrower['alt_labels'])
                prompt += f"Associated terms or alternative labels: {narrower_alt_labels_str}\n"
            
            prompt += "\n"
    
    return prompt

# Track if this is the first API call
first_api_call = True

def query_anthropic_with_retry(client, record_data, taxonomy_concepts, args, logger):
    """Query Anthropic API with retry logic."""
    retries = 0
    max_retries = args.max_retries
    delay = args.delay
    
    while retries <= max_retries:
        try:
            return query_anthropic_for_record(client, record_data, taxonomy_concepts, args, logger)
        except Exception as e:
            logger.error(f"Error for record {record_data['record_id']}: {e}")
            if retries < max_retries:
                wait_time = (2 ** retries) * delay
                logger.info(f"Retrying in {wait_time} seconds... ({retries+1}/{max_retries})")
                time.sleep(wait_time)
                retries += 1
            else:
                logger.error(f"Max retries reached for record {record_data['record_id']}")
                return None
    
    return None

def query_anthropic_for_record(client, record_data, taxonomy_concepts, args, logger):
    """Query Anthropic API for individual record classification."""
    global first_api_call
    
    # Prepare the record context
    context = f"Analyze the following catalog entry and determine its appropriate classification:\n\n"
    context += f"Person: {record_data.get('person', 'Unknown')}\n"
    context += f"Identity: {record_data.get('identity', 'Unknown')}\n"
    context += f"Composite: {record_data.get('composite', 'Unknown')}\n"
    
    # Include current classification if available
    if 'classification_label' in record_data and pd.notna(record_data['classification_label']):
        context += f"Current classification: {record_data['classification_label']}\n"
    if 'classification_path' in record_data and pd.notna(record_data['classification_path']):
        context += f"Current classification path: {record_data['classification_path']}\n"
    
    context += "\n"
    
    # Prepare taxonomy information
    taxonomy_prompt = prepare_taxonomy_prompt(taxonomy_concepts)
    
    # The prompt to the model
    prompt = f"""Based on the catalog entry provided, determine the appropriate classification for this specific record.

Your task is to:
1. Analyze the catalog entry (composite field, person name, and any other available information)
2. Review the current classification if provided
3. Use the following taxonomy categories to determine the most appropriate classification
4. Always choose a narrower category when determining the classification

{taxonomy_prompt}

5. Provide your final classification recommendation, which can include up to THREE concepts from the taxonomy
6. Provide a detailed rationale for this classification, explaining why it best represents the domain associated with this specific catalog entry

IMPORTANT GUIDELINES FOR CLASSIFICATION:
- The label values MUST come from the exact concept names in the taxonomy (preferred labels)
- You may assign a COMPOUND classification of up to THREE concepts when the documentary evidence strongly warrants it
- Only assign multiple classifications when there is clear evidence the record is associated with multiple domains
- The primary (first) classification should represent the most dominant domain for this record
- Single classifications are preferable when the evidence supports it
- You must ALWAYS provide a classification
- Focus on the specific evidence in THIS record, not general knowledge about the person

Please respond in JSON format with the following structure:
{{
  "label": ["Primary classification label", "Secondary classification label (if applicable)", "Tertiary classification label (if applicable)"],
  "path": ["Primary classification path", "Secondary classification path (if applicable)", "Tertiary classification path (if applicable)"],
  "rationale": "Detailed explanation for this classification, including justification for multiple classifications if assigned"
}}

The "path" field should include the full hierarchical path for each classification (e.g., "Arts, Culture, and Creative Expression > Visual Arts and Design").

Think step by step:
1. First, identify all available evidence about the subject matter from this specific catalog entry
2. Analyze the title, subject, and any other descriptive information
3. Determine which taxonomy category (or categories) best encompasses the domain represented by this record
4. Formulate your rationale, citing specific evidence from the catalog entry
"""

    # Log the full prompt for the first API call or if debug mode is enabled
    if first_api_call or args.debug:
        logger.info("=== SAMPLE API REQUEST ===")
        logger.info(f"Context:\n{context}")
        logger.info(f"Prompt:\n{prompt}")
        logger.info("=== END SAMPLE API REQUEST ===")
        first_api_call = False
    
    # Make the API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10000,
        temperature=0.0,
        system="You are an expert in library cataloging and classification who analyzes individual catalog entries to determine their appropriate classification category based on the subject matter and content described in the entry.",
        messages=[
            {"role": "user", "content": context + prompt}
        ]
    )
    
    # Extract and parse the JSON response
    result_text = response.content[0].text
    
    # Log the raw response for debugging
    if args.debug:
        logger.debug(f"=== RAW RESPONSE FOR RECORD {record_data['record_id']} ===")
        logger.debug(f"First 500 characters of response: {result_text[:500]}...")
        logger.debug("=== END RAW RESPONSE ===")
    
    result = extract_json_from_response(result_text, logger)
    
    if result:
        # Log parsed JSON for debugging
        if args.debug:
            logger.debug(f"=== PARSED JSON FOR RECORD {record_data['record_id']} ===")
            logger.debug(f"Full parsed result: {json.dumps(result, indent=2)}")
            logger.debug("=== END PARSED JSON ===")
    else:
        logger.error(f"Failed to extract JSON from response for record {record_data['record_id']}")
        if args.debug:
            logger.debug(f"Full raw response: {result_text}")
    
    return result

def process_records_in_batches(df, taxonomy_concepts, client, args, logger):
    """Process individual records in batches to manage API costs."""
    results = {}
    
    # Apply filters
    start_idx = args.start_index
    end_idx = len(df)
    
    if args.max_records:
        end_idx = min(start_idx + args.max_records, len(df))
    
    # Filter by specific record if specified
    if args.record_filter:
        try:
            record_idx = int(args.record_filter)
            if 0 <= record_idx < len(df):
                start_idx = record_idx
                end_idx = record_idx + 1
                logger.info(f"Filtering to process only record index: {record_idx}")
            else:
                logger.error(f"Record filter {record_idx} is out of range (0-{len(df)-1})")
                return results
        except ValueError:
            logger.error(f"Invalid record filter: {args.record_filter}. Must be an integer.")
            return results
    
    # Get the subset of records to process
    records_to_process = df.iloc[start_idx:end_idx]
    logger.info(f"Processing {len(records_to_process)} records (indices {start_idx} to {end_idx-1})")
    
    # Dry run check
    if args.dry_run:
        logger.info("DRY RUN MODE: No API calls will be made")
        for idx, record in records_to_process.head(3).iterrows():
            logger.info(f"Would process record {idx}: {record.get('person', 'Unknown')} - {record.get('composite', 'Unknown')[:100]}...")
        logger.info(f"Would process a total of {len(records_to_process)} records")
        return results
    
    # Process in batches
    for i in range(0, len(records_to_process), args.batch_size):
        batch = records_to_process.iloc[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(records_to_process) + args.batch_size - 1) // args.batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        for idx, record in tqdm(batch.iterrows(), desc="Processing records", total=len(batch)):
            logger.info(f"Processing record {idx}")
            
            # Convert record to dict for easier handling
            record_data = record.to_dict()
            record_data['record_id'] = idx
            
            # Query Anthropic API with retry
            result = query_anthropic_with_retry(client, record_data, taxonomy_concepts, args, logger)
            
            if result:
                # Store the result with the record index as key
                results[str(idx)] = {
                    'record_id': idx,
                    'person': record_data.get('person', 'Unknown'),
                    'identity': record_data.get('identity', 'Unknown'),
                    'composite': record_data.get('composite', 'Unknown'),
                    'original_classification_label': record_data.get('classification_label', None),
                    'original_classification_path': record_data.get('classification_path', None),
                    'new_classification_label': result['label'],
                    'new_classification_path': result['path'],
                    'rationale': result['rationale']
                }
                
                # Log the classification(s) for verification
                if isinstance(result['label'], list):
                    label_str = ", ".join(result['label'])
                else:
                    label_str = result['label']
                logger.info(f"Classified record {idx}: {label_str}")
                
                # If multiple classifications were assigned, log this information
                if isinstance(result['label'], list) and len(result['label']) > 1:
                    logger.info(f"Multiple classifications ({len(result['label'])}) assigned to record {idx}")
            else:
                logger.warning(f"Failed to get classification for record {idx}")
            
            # Add a delay to respect API rate limits
            time.sleep(args.delay)
        
        # Save progress after each batch
        logger.info(f"Saving progress after batch {batch_num}...")
        with open(f"{args.output}.partial", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final reporting
    logger.info(f"Processed {len(results)} records")
    
    # Count and report on compound classifications
    compound_count = 0
    for record_id, data in results.items():
        if isinstance(data.get('new_classification_label'), list) and len(data.get('new_classification_label', [])) > 1:
            compound_count += 1
    
    if compound_count > 0:
        logger.info(f"Found {compound_count} compound classifications ({compound_count/len(results)*100:.1f}% of processed records)")
    
    return results

def main():
    """Main function to run the script."""
    args = parse_arguments()
    logger = setup_logging()
    
    # Adjust log level if debug mode is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - increased logging verbosity")
    
    logger.info("Starting individual record classification verification script")
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("No Anthropic API key provided. Use --api-key or set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Initialize Anthropic client (only if not in dry run mode)
    client = None
    if not args.dry_run:
        client = Anthropic(api_key=api_key)
        logger.info("Initialized Anthropic client with model claude-sonnet-4")
    
    # Load data
    logger.info("Loading data...")
    df, taxonomy_concepts = load_data(args, logger)
    
    # Log taxonomy concepts info
    logger.info(f"Loaded taxonomy with {len(taxonomy_concepts)} top-level concepts")
    for concept_id, concept in taxonomy_concepts.items():
        logger.info(f"  - {concept['label']} with {len(concept['narrower'])} sub-concepts")
    
    # Process records in batches
    results = process_records_in_batches(df, taxonomy_concepts, client, args, logger)
    
    # Write final output (unless in dry run mode)
    if not args.dry_run:
        logger.info("Writing classification results to file...")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log output file info
        output_size = os.path.getsize(args.output)
        logger.info(f"Output file size: {output_size} bytes")
        
        # Remove partial file if it exists
        if os.path.exists(f"{args.output}.partial"):
            os.remove(f"{args.output}.partial")
        
        logger.info(f"Done! Individual record classification results written to {args.output}")
        logger.info(f"Summary: {len(results)} records processed and classified")
    else:
        logger.info("Dry run completed. No files were written.")

if __name__ == "__main__":
    main()