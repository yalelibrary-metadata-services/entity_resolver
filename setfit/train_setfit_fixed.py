#!/usr/bin/env python3
"""
SetFit Entity Classification CLI - Simple Fixed Version

Optimized for A10G GPU with proper SetFit TrainingArguments.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class SimpleHierarchicalClassifier:
    """Simple hierarchical classifier optimized for GPU training."""
    
    def __init__(self, min_examples_threshold=8):
        self.min_examples_threshold = min_examples_threshold
        self.label_mapping = {}
        self.parent_mapping = {}
        self.model = None
        self.rare_classes = set()
        
    def create_label_hierarchy(self, df):
        """Create hierarchical label mapping based on class counts."""
        # Store parent mapping
        for idx, row in df.iterrows():
            self.parent_mapping[row['label']] = row['parent_category']
        
        # Count examples per class
        class_counts = df['label'].value_counts()
        
        # Determine which classes need parent fallback
        print("\nHierarchical mapping:")
        for label, count in class_counts.items():
            if count < self.min_examples_threshold:
                parent = self.parent_mapping[label]
                self.label_mapping[label] = parent
                self.rare_classes.add(label)
                print(f"  {label} ({count} examples) -> {parent}")
            else:
                self.label_mapping[label] = label
        
        # Create new column with hierarchical labels
        df['hierarchical_label'] = df['label'].map(self.label_mapping)
        
        print(f"\nTotal classes affected: {len(self.rare_classes)}")
        return df
    
    def train(self, train_df, val_df, model_name="sentence-transformers/paraphrase-mpnet-base-v2",
              batch_size=32, num_epochs=3):
        """Train SetFit model with optimized settings."""
        
        print(f"Training with batch_size={batch_size}, epochs={num_epochs}")
        
        # Apply hierarchical mapping
        train_df = self.create_label_hierarchy(train_df.copy())
        val_df = val_df.copy()
        val_df['hierarchical_label'] = val_df['label'].map(self.label_mapping)
        
        # Create datasets
        train_dataset = Dataset.from_pandas(
            train_df[['composite', 'hierarchical_label']].rename(
                columns={'composite': 'text', 'hierarchical_label': 'label'}
            )
        )
        
        val_dataset = Dataset.from_pandas(
            val_df[['composite', 'hierarchical_label']].rename(
                columns={'composite': 'text', 'hierarchical_label': 'label'}
            )
        )
        
        # Get unique labels
        unique_labels = sorted(train_df['hierarchical_label'].unique())
        print(f"\nTraining with {len(unique_labels)} unique labels")
        
        # Initialize model
        self.model = SetFitModel.from_pretrained(
            model_name,
            labels=unique_labels
        )
        
        # Simple training arguments that work with SetFit
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            output_dir="./setfit_output",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            metric="accuracy"
        )
        
        # Train with timing
        print(f"\nStarting training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return trainer
    
    def predict(self, texts):
        """Make predictions using the trained model."""
        return self.model.predict(texts)
    
    def evaluate_dual(self, test_df):
        """Evaluate at both hierarchical and original levels."""
        predictions = self.predict(test_df['composite'].tolist())
        
        test_df['hierarchical_label'] = test_df['label'].map(self.label_mapping)
        hier_accuracy = accuracy_score(test_df['hierarchical_label'], predictions)
        orig_accuracy = accuracy_score(test_df['label'], predictions)
        
        print("\n" + "="*50)
        print("DUAL EVALUATION RESULTS")
        print("="*50)
        print(f"Hierarchical Accuracy: {hier_accuracy:.3f}")
        print(f"Original Accuracy: {orig_accuracy:.3f}")
        print(f"Granularity Loss: {hier_accuracy - orig_accuracy:.3f}")
        
        return {
            'hierarchical_accuracy': hier_accuracy,
            'original_accuracy': orig_accuracy,
            'predictions': predictions,
            'test_df': test_df
        }
    
    def save_model(self, output_dir):
        """Save the trained model and mappings."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / "setfit_model"
        self.model.save_pretrained(str(model_path))
        
        metadata = {
            'label_mapping': self.label_mapping,
            'parent_mapping': self.parent_mapping,
            'rare_classes': list(self.rare_classes),
            'min_examples_threshold': self.min_examples_threshold
        }
        
        with open(output_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {output_path}")


def load_entity_data(csv_path, ground_truth_path):
    """Load entity data and merge with ground truth labels."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entities from CSV")
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    identity_to_label = {}
    identity_to_path = {}
    
    for identity, info in ground_truth.items():
        if 'label' in info and len(info['label']) > 0:
            identity_to_label[identity] = info['label'][0]
            if 'path' in info and len(info['path']) > 0:
                identity_to_path[identity] = info['path'][0]
    
    df['label'] = df['identity'].astype(str).map(identity_to_label)
    df['path'] = df['identity'].astype(str).map(identity_to_path)
    
    df_labeled = df.dropna(subset=['label']).copy()
    print(f"Found labels for {len(df_labeled)} entities")
    
    df_labeled['parent_category'] = df_labeled['path'].apply(
        lambda x: x.split(' > ')[0] if pd.notna(x) else None
    )
    
    return df_labeled


def main():
    parser = argparse.ArgumentParser(description='Simple SetFit training for A10G')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with entity data')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to JSON file with ground truth labels')
    parser.add_argument('--output_dir', type=str, default='./setfit_model_output',
                        help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, 
                        default='sentence-transformers/paraphrase-mpnet-base-v2',
                        help='Pre-trained model name')
    parser.add_argument('--min_examples', type=int, default=8,
                        help='Minimum examples per class')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (try 16, 24, 32, or 48)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set proportion')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("No GPU detected, using CPU")
    
    # Load data
    print("Loading data...")
    df = load_entity_data(args.csv_path, args.ground_truth_path)
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df['label'], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=args.val_size, stratify=train_val_df['label'], random_state=args.seed
    )
    
    print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Train
    classifier = SimpleHierarchicalClassifier(min_examples_threshold=args.min_examples)
    trainer = classifier.train(
        train_df, val_df, 
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    # Evaluate
    print("\nEvaluating...")
    results = classifier.evaluate_dual(test_df)
    
    # Save
    classifier.save_model(args.output_dir)
    
    print(f"\nFinal results:")
    print(f"Hierarchical accuracy: {results['hierarchical_accuracy']:.3f}")
    print(f"Original accuracy: {results['original_accuracy']:.3f}")


if __name__ == "__main__":
    main()