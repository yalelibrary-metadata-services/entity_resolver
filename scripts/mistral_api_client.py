#!/usr/bin/env python3
"""
Mistral API client for entity taxonomy classification.

This script provides a command-line interface to interact with Mistral's 
Classifier Factory API for training and inference.
"""

import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from mistralai import Mistral


class MistralEntityClassifier:
    """Client for Mistral Classifier Factory API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Mistral client."""
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=self.api_key)
        print(f"Initialized Mistral client")
    
    def upload_file(self, file_path: str) -> str:
        """Upload a file to Mistral and return file ID."""
        print(f"Uploading {file_path} to Mistral...")
        
        with open(file_path, "rb") as f:
            uploaded_file = self.client.files.upload(file=f)
        
        file_id = uploaded_file.id
        print(f"✓ Uploaded successfully. File ID: {file_id}")
        return file_id
    
    def create_training_job(self, train_file_id: str, val_file_id: str, 
                          job_name: str = "entity-taxonomy-classifier") -> str:
        """Create a fine-tuning job."""
        print(f"Creating training job '{job_name}'...")
        
        job = self.client.fine_tuning.jobs.create(
            model="ministral-3b-latest",
            training_files=[{"file_id": train_file_id, "weight": 1}],
            validation_files=[val_file_id],
            hyperparameters={
                "training_steps": 100,  # Adjust based on data size
                "learning_rate": 0.0001,
            },
            auto_start=True,
        )
        
        job_id = job.id
        print(f"✓ Training job created: {job_id}")
        print(f"  Status: {job.status}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a training job."""
        job = self.client.fine_tuning.jobs.get(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": getattr(job, 'fine_tuned_model', None),
            "message": getattr(job, 'message', ''),
            "created_at": getattr(job, 'created_at', ''),
        }
    
    def monitor_training(self, job_id: str, check_interval: int = 60) -> Optional[str]:
        """Monitor training progress and return model ID when complete."""
        print(f"Monitoring training job {job_id}...")
        print(f"Checking every {check_interval} seconds...")
        
        while True:
            status_info = self.get_job_status(job_id)
            status = status_info["status"]
            
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
            
            if status == "SUCCESS":
                model_id = status_info["model"]
                print(f"🎉 Training completed successfully!")
                print(f"   Model ID: {model_id}")
                return model_id
            
            elif status == "FAILED":
                print(f"❌ Training failed: {status_info.get('message', 'Unknown error')}")
                return None
            
            elif status in ["RUNNING", "QUEUED", "VALIDATING"]:
                print(f"   Training in progress...")
                time.sleep(check_interval)
            
            else:
                print(f"   Unknown status: {status}")
                time.sleep(check_interval)
    
    def classify_text(self, text: str, model_id: str) -> Dict:
        """Classify a single text using the fine-tuned model."""
        try:
            response = self.client.classifiers.classify(
                model=model_id,
                inputs=[text]
            )
            
            result = response.results[0]
            classification = {}
            
            # Extract predictions for each label
            for label_name, prediction in result.predictions.items():
                if hasattr(prediction, 'value'):
                    classification[label_name] = prediction.value
                else:
                    classification[label_name] = str(prediction)
            
            return classification
        
        except Exception as e:
            print(f"Error classifying text: {e}")
            return {}
    
    def classify_batch(self, texts: List[str], model_id: str, batch_size: int = 10) -> List[Dict]:
        """Classify multiple texts in batches."""
        print(f"Classifying {len(texts)} texts in batches of {batch_size}...")
        
        all_classifications = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            batch_results = []
            for text in batch:
                classification = self.classify_text(text, model_id)
                batch_results.append(classification)
                time.sleep(0.1)  # Rate limiting
            
            all_classifications.extend(batch_results)
        
        print(f"✓ Completed classification of {len(all_classifications)} texts")
        return all_classifications


def train_workflow(args):
    """Complete training workflow."""
    classifier = MistralEntityClassifier(args.api_key)
    
    # Upload files
    train_file_id = classifier.upload_file(args.train_file)
    val_file_id = classifier.upload_file(args.val_file)
    
    # Create training job
    job_id = classifier.create_training_job(
        train_file_id, 
        val_file_id,
        job_name=args.job_name
    )
    
    # Monitor training
    model_id = classifier.monitor_training(job_id, args.check_interval)
    
    if model_id:
        # Save model ID
        model_id_path = Path(args.output_dir) / "mistral_model_id.txt"
        model_id_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_id_path, 'w') as f:
            f.write(model_id)
        
        print(f"✓ Model ID saved to: {model_id_path}")
        
        # Save training info
        training_info = {
            "job_id": job_id,
            "model_id": model_id,
            "train_file_id": train_file_id,
            "val_file_id": val_file_id,
            "job_name": args.job_name,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        info_path = Path(args.output_dir) / "mistral_training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✓ Training info saved to: {info_path}")
    
    return model_id


def classify_workflow(args):
    """Classification workflow."""
    classifier = MistralEntityClassifier(args.api_key)
    
    # Load model ID
    if args.model_id:
        model_id = args.model_id
    else:
        model_id_path = Path(args.model_id_file)
        if not model_id_path.exists():
            raise FileNotFoundError(f"Model ID file not found: {model_id_path}")
        
        with open(model_id_path, 'r') as f:
            model_id = f.read().strip()
    
    print(f"Using model: {model_id}")
    
    if args.text:
        # Single text classification
        print(f"\nClassifying text: {args.text[:100]}...")
        classification = classifier.classify_text(args.text, model_id)
        
        print(f"\nClassification result:")
        for label, value in classification.items():
            print(f"  {label}: {value}")
    
    elif args.input_csv:
        # Batch classification from CSV
        print(f"\nLoading data from {args.input_csv}...")
        df = pd.read_csv(args.input_csv)
        
        if args.text_column not in df.columns:
            raise ValueError(f"Column '{args.text_column}' not found in CSV")
        
        texts = df[args.text_column].tolist()
        
        # Classify
        classifications = classifier.classify_batch(texts, model_id, args.batch_size)
        
        # Process results
        for i, classification in enumerate(classifications):
            domains = classification.get('domain', [])
            parents = classification.get('parent_category', [])
            
            df.at[i, 'mistral_domains'] = '|'.join(domains) if isinstance(domains, list) else domains
            df.at[i, 'mistral_parent_categories'] = '|'.join(parents) if isinstance(parents, list) else parents
            df.at[i, 'mistral_num_domains'] = len(domains) if isinstance(domains, list) else 1
            df.at[i, 'mistral_primary_domain'] = domains[0] if domains else ''
        
        # Save results
        output_path = args.output_csv or args.input_csv.replace('.csv', '_mistral_classified.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
        
        # Show summary
        print(f"\nClassification summary:")
        if 'mistral_primary_domain' in df.columns:
            domain_counts = df['mistral_primary_domain'].value_counts()
            for domain, count in domain_counts.head(10).items():
                print(f"  {domain}: {count}")


def status_workflow(args):
    """Check job status."""
    classifier = MistralEntityClassifier(args.api_key)
    status_info = classifier.get_job_status(args.job_id)
    
    print(f"Job Status for {args.job_id}:")
    for key, value in status_info.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Mistral API client for entity taxonomy classification')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--api_key', type=str, help='Mistral API key (or set MISTRAL_API_KEY env var)')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[parent_parser], help='Train a new classifier')
    train_parser.add_argument('--train_file', type=str, required=True, help='Path to training JSONL file')
    train_parser.add_argument('--val_file', type=str, required=True, help='Path to validation JSONL file')
    train_parser.add_argument('--job_name', type=str, default='entity-taxonomy-classifier', help='Training job name')
    train_parser.add_argument('--output_dir', type=str, default='data/output', help='Output directory for model info')
    train_parser.add_argument('--check_interval', type=int, default=60, help='Status check interval (seconds)')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', parents=[parent_parser], help='Classify text(s)')
    classify_parser.add_argument('--model_id', type=str, help='Model ID (if not provided, read from file)')
    classify_parser.add_argument('--model_id_file', type=str, default='data/output/mistral_model_id.txt', 
                                help='Path to file containing model ID')
    classify_parser.add_argument('--text', type=str, help='Single text to classify')
    classify_parser.add_argument('--input_csv', type=str, help='CSV file with texts to classify')
    classify_parser.add_argument('--text_column', type=str, default='composite', help='Column name containing text')
    classify_parser.add_argument('--output_csv', type=str, help='Output CSV path (default: input_classified.csv)')
    classify_parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    
    # Status command
    status_parser = subparsers.add_parser('status', parents=[parent_parser], help='Check training job status')
    status_parser.add_argument('--job_id', type=str, required=True, help='Training job ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_workflow(args)
        elif args.command == 'classify':
            classify_workflow(args)
        elif args.command == 'status':
            status_workflow(args)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())