#!/usr/bin/env python3
"""
SetFit Entity Classification CLI

Hierarchical text classification using SetFit for entity resolution with imbalanced classes.
Trains on preprocessed ground-truth data with broader class labels substituted for sparse ones.
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class HierarchicalClassifier:
    """
    A hierarchical classifier that falls back to parent categories
    for underrepresented classes.
    """
    
    def __init__(self, min_examples_threshold=8):
        self.min_examples_threshold = min_examples_threshold
        self.label_mapping = {}
        self.parent_mapping = {}
        self.model = None
        self.rare_classes = set()
        
    def create_label_hierarchy(self, df):
        """
        Create hierarchical label mapping based on class counts.
        """
        # Store parent mapping
        for idx, row in df.iterrows():
            self.parent_mapping[row['label']] = row['parent_category']
        
        # Count examples per class
        class_counts = df['label'].value_counts()
        
        # Determine which classes need parent fallback
        print("\nHierarchical mapping:")
        for label, count in class_counts.items():
            if count < self.min_examples_threshold:
                # Use parent category
                parent = self.parent_mapping[label]
                self.label_mapping[label] = parent
                self.rare_classes.add(label)
                print(f"  {label} ({count} examples) -> {parent}")
            else:
                # Use original label
                self.label_mapping[label] = label
        
        # Create new column with hierarchical labels
        df['hierarchical_label'] = df['label'].map(self.label_mapping)
        
        print(f"\nTotal classes affected: {len(self.rare_classes)}")
        return df
    
    def train(self, train_df, val_df, model_name="sentence-transformers/paraphrase-mpnet-base-v2"):
        """
        Train SetFit model with hierarchical labels.
        """
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
        print(f"\nTraining with {len(unique_labels)} unique labels: {unique_labels}")
        
        # Initialize model
        self.model = SetFitModel.from_pretrained(
            model_name,
            labels=unique_labels
        )
        
        # Training arguments
        args = TrainingArguments(
            batch_size=16,
            num_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            output_dir="./setfit_output",
            logging_dir="./setfit_logs"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            metric="accuracy"
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        return trainer
    
    def predict(self, texts):
        """Make predictions using the trained model."""
        return self.model.predict(texts)
    
    def evaluate_dual(self, test_df):
        """
        Evaluate at both hierarchical and original levels.
        """
        # Get predictions
        predictions = self.predict(test_df['composite'].tolist())
        
        # Hierarchical evaluation
        test_df['hierarchical_label'] = test_df['label'].map(self.label_mapping)
        hier_accuracy = accuracy_score(test_df['hierarchical_label'], predictions)
        
        # Original evaluation
        orig_accuracy = accuracy_score(test_df['label'], predictions)
        
        print("\n" + "="*50)
        print("DUAL EVALUATION RESULTS")
        print("="*50)
        print(f"\nHierarchical Accuracy: {hier_accuracy:.3f}")
        print(f"  (How well we predict at the level we trained for)")
        print(f"\nOriginal Accuracy: {orig_accuracy:.3f}")
        print(f"  (How well we predict the exact category)")
        print(f"\nGranularity Loss: {hier_accuracy - orig_accuracy:.3f}")
        print(f"  (The cost of using parent categories)")
        
        # Classification report
        print("\nClassification Report (Hierarchical Level):")
        print(classification_report(test_df['hierarchical_label'], predictions))
        
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
        
        # Save the SetFit model
        model_path = output_path / "setfit_model"
        self.model.save_pretrained(str(model_path))
        print(f"Model saved to {model_path}")
        
        # Save mappings and metadata
        metadata = {
            'label_mapping': self.label_mapping,
            'parent_mapping': self.parent_mapping,
            'rare_classes': list(self.rare_classes),
            'min_examples_threshold': self.min_examples_threshold
        }
        
        metadata_path = output_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        
        # Save metadata as JSON for readability
        metadata_json = metadata.copy()
        metadata_json['rare_classes'] = list(metadata_json['rare_classes'])
        
        json_path = output_path / "metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        print(f"Metadata (JSON) saved to {json_path}")


def load_entity_data(csv_path, ground_truth_path):
    """
    Load entity data and merge with ground truth labels.
    
    Args:
        csv_path: Path to the CSV file with entity data
        ground_truth_path: Path to the JSON file with ground truth labels
    
    Returns:
        DataFrame with entities and their labels
    """
    # Load the main dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entities from CSV")
    
    # Load ground truth labels
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Create a mapping from identity to label (using only primary label)
    identity_to_label = {}
    identity_to_path = {}
    
    for identity, info in ground_truth.items():
        if 'label' in info and len(info['label']) > 0:
            # Take only the primary (first) label
            identity_to_label[identity] = info['label'][0]
            # Also store the path for hierarchical classification
            if 'path' in info and len(info['path']) > 0:
                identity_to_path[identity] = info['path'][0]
    
    # Merge labels with dataset
    df['label'] = df['identity'].astype(str).map(identity_to_label)
    df['path'] = df['identity'].astype(str).map(identity_to_path)
    
    # Filter out rows without labels
    df_labeled = df.dropna(subset=['label'])
    print(f"Found labels for {len(df_labeled)} entities")
    
    # Extract parent category from path
    df_labeled['parent_category'] = df_labeled['path'].apply(
        lambda x: x.split(' > ')[0] if pd.notna(x) else None
    )
    
    return df_labeled


def analyze_class_distribution(df):
    """Analyze and print class distribution statistics."""
    class_counts = df['label'].value_counts()
    
    print(f"\nTotal number of classes: {len(class_counts)}")
    print(f"Total number of examples: {len(df)}")
    print(f"Average examples per class: {len(df) / len(class_counts):.1f}")
    print(f"\nClasses with < 8 examples: {sum(class_counts < 8)} ({sum(class_counts < 8) / len(class_counts) * 100:.1f}%)")
    print(f"Classes with < 16 examples: {sum(class_counts < 16)} ({sum(class_counts < 16) / len(class_counts) * 100:.1f}%)")
    
    print("\nClass distribution (showing smallest classes):")
    print(class_counts.tail(10))
    
    # Analyze parent category distribution
    parent_counts = df['parent_category'].value_counts()
    print(f"\nParent category distribution:")
    for parent, count in parent_counts.items():
        print(f"  {parent}: {count} examples")


def main():
    parser = argparse.ArgumentParser(description='Train SetFit classifier for entity resolution')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with entity data')
    parser.add_argument('--ground_truth_path', type=str, required=True,
                        help='Path to JSON file with ground truth labels')
    parser.add_argument('--output_dir', type=str, default='./setfit_model_output',
                        help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, 
                        default='sentence-transformers/paraphrase-mpnet-base-v2',
                        help='Pre-trained model name from HuggingFace')
    parser.add_argument('--min_examples', type=int, default=8,
                        help='Minimum examples per class for direct classification')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, mps, cpu, or auto)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = load_entity_data(args.csv_path, args.ground_truth_path)
    
    # Analyze class distribution
    analyze_class_distribution(df)
    
    # Split data
    print(f"\nSplitting data (test_size={args.test_size}, val_size={args.val_size})...")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=args.test_size, 
        stratify=df['label'],
        random_state=args.seed
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_size,
        stratify=train_val_df['label'],
        random_state=args.seed
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Initialize and train classifier
    print(f"\nInitializing hierarchical classifier (min_examples={args.min_examples})...")
    classifier = HierarchicalClassifier(min_examples_threshold=args.min_examples)
    
    # Modify training arguments based on CLI args
    trainer = classifier.train(train_df, val_df, model_name=args.model_name)
    
    # Evaluate
    print("\nEvaluating model...")
    results = classifier.evaluate_dual(test_df)
    
    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    classifier.save_model(args.output_dir)
    
    # Save evaluation results
    results_path = Path(args.output_dir) / "evaluation_results.json"
    eval_results = {
        'hierarchical_accuracy': results['hierarchical_accuracy'],
        'original_accuracy': results['original_accuracy'],
        'granularity_loss': results['hierarchical_accuracy'] - results['original_accuracy'],
        'test_size': len(test_df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'total_classes': len(df['label'].unique()),
        'rare_classes_count': len(classifier.rare_classes)
    }
    
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")
    
    print("\nTraining completed successfully!")
    print(f"Final hierarchical accuracy: {results['hierarchical_accuracy']:.3f}")
    print(f"Final original accuracy: {results['original_accuracy']:.3f}")


if __name__ == "__main__":
    main()