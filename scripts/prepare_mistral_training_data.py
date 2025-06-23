#!/usr/bin/env python3
"""
Prepare training data for Mistral Classifier Factory from parallel classifications.

This script combines composite text from training_dataset.csv with multi-label 
classifications from parallel_classifications.json to create Mistral-compatible JSONL format.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import random


def load_composite_data(csv_path: str) -> dict:
    """Load composite text data keyed by personId."""
    print(f"Loading composite data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    composite_lookup = {}
    
    for _, row in df.iterrows():
        person_id = str(row['personId'])
        composite = row['composite']
        composite_lookup[person_id] = composite
    
    print(f"Loaded composite text for {len(composite_lookup)} unique personIds")
    return composite_lookup


def load_parallel_classifications(json_path: str) -> dict:
    """Load parallel classifications data."""
    print(f"Loading parallel classifications from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        classifications = json.load(f)
    
    print(f"Loaded classifications for {len(classifications)} entities")
    return classifications


def create_mistral_training_examples(composite_lookup: dict, classifications: dict) -> list:
    """Create Mistral-compatible training examples."""
    training_examples = []
    skipped_no_composite = 0
    skipped_no_labels = 0
    
    for person_id, classification_data in classifications.items():
        # Get composite text
        composite_text = composite_lookup.get(person_id)
        if not composite_text:
            skipped_no_composite += 1
            continue
        
        # Extract labels and paths
        labels_list = classification_data.get('label', [])
        paths_list = classification_data.get('path', [])
        
        if not labels_list:
            skipped_no_labels += 1
            continue
        
        # Extract parent categories from paths
        parent_categories = []
        for path in paths_list:
            if " > " in path:
                parent_categories.append(path.split(" > ")[0])
        
        # Create Mistral labels using native multi-label format
        mistral_labels = {
            "domain": labels_list,  # List of domains (native multi-label)
            "parent_category": parent_categories  # List of parent categories
        }
        
        training_examples.append({
            "text": composite_text,
            "labels": mistral_labels
        })
    
    print(f"Created {len(training_examples)} training examples")
    print(f"Skipped {skipped_no_composite} entries (no composite text)")
    print(f"Skipped {skipped_no_labels} entries (no labels)")
    
    return training_examples


def analyze_label_distribution(training_examples: list):
    """Analyze and report label distribution."""
    single_domain = 0
    dual_domain = 0
    triple_domain = 0
    parent_spans = 0
    
    domain_counts = {}
    parent_counts = {}
    
    for example in training_examples:
        domains = example["labels"]["domain"]
        parents = example["labels"]["parent_category"]
        
        # Count domain multiplicity
        num_domains = len(domains)
        if num_domains == 1:
            single_domain += 1
        elif num_domains == 2:
            dual_domain += 1
        elif num_domains == 3:
            triple_domain += 1
        
        # Count parent category spans
        unique_parents = set(parents)
        if len(unique_parents) > 1:
            parent_spans += 1
        
        # Count individual domains and parents
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        for parent in unique_parents:
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
    
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"Multi-label distribution:")
    print(f"  Single domain:  {single_domain:4d} ({single_domain/len(training_examples)*100:.1f}%)")
    print(f"  Dual domain:    {dual_domain:4d} ({dual_domain/len(training_examples)*100:.1f}%)")
    print(f"  Triple domain:  {triple_domain:4d} ({triple_domain/len(training_examples)*100:.1f}%)")
    print(f"  Multi-parent:   {parent_spans:4d} ({parent_spans/len(training_examples)*100:.1f}%)")
    
    print(f"\nTop 10 domains:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {domain:<50} {count:4d}")
    
    print(f"\nParent categories:")
    for parent, count in sorted(parent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {parent:<50} {count:4d}")


def split_train_validation(training_examples: list, val_ratio: float = 0.2, random_seed: int = 42):
    """Split training examples into train and validation sets."""
    random.seed(random_seed)
    
    # Shuffle the data
    shuffled_examples = training_examples.copy()
    random.shuffle(shuffled_examples)
    
    # Split
    split_idx = int(len(shuffled_examples) * (1 - val_ratio))
    train_examples = shuffled_examples[:split_idx]
    val_examples = shuffled_examples[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training:   {len(train_examples):4d} examples")
    print(f"  Validation: {len(val_examples):4d} examples")
    
    return train_examples, val_examples


def write_jsonl(examples: list, output_path: str):
    """Write examples to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Mistral training data from parallel classifications')
    parser.add_argument('--composite_csv', type=str, 
                        default='data/input/training_dataset.csv',
                        help='Path to CSV with composite text data')
    parser.add_argument('--classifications_json', type=str,
                        default='data/input/parallel_classifications.json', 
                        help='Path to JSON with parallel classifications')
    parser.add_argument('--output_dir', type=str,
                        default='data/output',
                        help='Output directory for JSONL files')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for train/val split')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    composite_lookup = load_composite_data(args.composite_csv)
    classifications = load_parallel_classifications(args.classifications_json)
    
    # Create training examples
    training_examples = create_mistral_training_examples(composite_lookup, classifications)
    
    if not training_examples:
        print("ERROR: No training examples created!")
        return
    
    # Analyze distribution
    analyze_label_distribution(training_examples)
    
    # Split train/validation
    train_examples, val_examples = split_train_validation(
        training_examples, 
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    
    # Write JSONL files
    train_path = output_dir / "mistral_train.jsonl"
    val_path = output_dir / "mistral_val.jsonl"
    full_path = output_dir / "mistral_full_dataset.jsonl"
    
    write_jsonl(train_examples, str(train_path))
    write_jsonl(val_examples, str(val_path))
    write_jsonl(training_examples, str(full_path))
    
    print("\n" + "="*60)
    print("SUCCESS! Mistral training data prepared.")
    print("="*60)
    print(f"Files created:")
    print(f"  Training:   {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Full set:   {full_path}")
    print()
    print("Next steps:")
    print("1. Upload train and validation files to Mistral")
    print("2. Create fine-tuning job")
    print("3. Monitor training progress")


if __name__ == "__main__":
    main()