#!/usr/bin/env python3
"""
Spurious Correlations Experiment (inspired by Risse et al. 2025)

Tests whether datasets can be exploited using only word counts (bag-of-words)
with a simple Gradient Boosting Classifier, completely ignoring code structure.

This helps identify which datasets contain spurious features that allow
high performance without actually understanding vulnerabilities.
"""

import os
import random
import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer


# Dataset configurations - maps dataset names to their file paths and label columns
DATASET_CONFIGS = {
    'devign': {
        'path': 'data/devign/devign_splits/train.jsonl',
        'test_path': 'data/devign/devign_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'dedup-devign': {
        'path': 'data/dedup-devign/dedup-devign_splits/train.jsonl',
        'test_path': 'data/dedup-devign/dedup-devign_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'bigvul': {
        'path': 'data/bigvul/bigvul_splits/train.jsonl',
        'test_path': 'data/bigvul/bigvul_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'dedup-bigvul': {
        'path': 'data/dedup-bigvul/dedup-bigvul_splits/train.jsonl',
        'test_path': 'data/dedup-bigvul/dedup-bigvul_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'diversevul': {
        'path': 'data/diversevul/diversevul_splits/train.jsonl',
        'test_path': 'data/diversevul/diversevul_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'cvefixes': {
        'path': 'data/cvefixes/cvefixes_splits/train.jsonl',
        'test_path': 'data/cvefixes/cvefixes_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'draper': {
        'path': 'data/draper/draper_splits/train.jsonl',
        'test_path': 'data/draper/draper_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'icvul': {
        'path': 'data/icvul/icvul_splits/train.jsonl',
        'test_path': 'data/icvul/icvul_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'juliet': {
        'path': 'data/juliet/juliet_splits/train.jsonl',
        'test_path': 'data/juliet/juliet_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'dedup-juliet': {
        'path': 'data/dedup-juliet/dedup-juliet_splits/train.jsonl',
        'test_path': 'data/dedup-juliet/dedup-juliet_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'reveal': {
        'path': 'data/reveal/reveal_splits/train.jsonl',
        'test_path': 'data/reveal/reveal_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    },
    'vuldeepecker': {
        'path': 'data/vuldeepecker/vuldeepecker_splits/train.jsonl',
        'test_path': 'data/vuldeepecker/vuldeepecker_splits/test.jsonl',
        'func_col': 'func',
        'label_col': 'target'
    }
}


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_jsonl(file_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def encode_dataframe(data, func_col, label_col, tokenizer, max_samples=None):
    """
    Encode functions as word count vectors.

    Args:
        data: List of dicts with function code and labels
        func_col: Column name for function code
        label_col: Column name for labels
        tokenizer: HuggingFace tokenizer
        max_samples: Optional limit on number of samples (for testing)

    Returns:
        X: List of word count vectors
        y: List of labels
    """
    vocabulary = tokenizer.get_vocab()
    max_vocab_index = max(vocabulary.values())

    word_counts = []
    labels = []

    samples = data[:max_samples] if max_samples else data

    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(samples)} samples")

        func_text = sample.get(func_col, '')
        label = sample.get(label_col, 0)

        # Tokenize and get word counts
        tokens = tokenizer.tokenize(str(func_text))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        count = Counter(token_ids)

        # Create count vector
        count_vector = [0] * (max_vocab_index + 1)
        for idx, cnt in count.items():
            if idx <= max_vocab_index:
                count_vector[idx] = cnt

        word_counts.append(count_vector)
        labels.append(label)

    return word_counts, labels


def run_experiment(dataset_name, project_root, tokenizer, max_samples=None, seed=42):
    """
    Run spurious correlation experiment on a single dataset.

    Args:
        dataset_name: Name of dataset to test
        project_root: Path to VulnBench root directory
        tokenizer: HuggingFace tokenizer
        max_samples: Optional limit on samples (for quick testing)
        seed: Random seed

    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment on: {dataset_name}")
    print(f"{'='*60}")

    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        print(f"ERROR: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return None

    config = DATASET_CONFIGS[dataset_name]

    # Build full paths
    train_path = Path(project_root) / config['path']
    test_path = Path(project_root) / config['test_path']

    # Check if files exist
    if not train_path.exists():
        print(f"ERROR: Train file not found: {train_path}")
        return None
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}")
        return None

    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")

    # Load data
    print("Loading data...")
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    print(f"Loaded {len(train_data)} train samples")
    print(f"Loaded {len(test_data)} test samples")

    # Encode as word counts
    print("Encoding training data as word counts...")
    X_train, y_train = encode_dataframe(
        train_data,
        config['func_col'],
        config['label_col'],
        tokenizer,
        max_samples=max_samples
    )

    print("Encoding test data as word counts...")
    X_test, y_test = encode_dataframe(
        test_data,
        config['func_col'],
        config['label_col'],
        tokenizer,
        max_samples=max_samples
    )

    # Calculate class distribution
    train_pos = sum(y_train)
    test_pos = sum(y_test)
    print(f"\nClass distribution:")
    print(f"  Train: {train_pos}/{len(y_train)} ({100*train_pos/len(y_train):.1f}% vulnerable)")
    print(f"  Test:  {test_pos}/{len(y_test)} ({100*test_pos/len(y_test):.1f}% vulnerable)")

    # Train gradient boosting classifier
    # Using same hyperparameters as Risse et al.
    print("\nTraining Gradient Boosting Classifier...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.3,
        max_depth=10,
        max_iter=200,
        min_samples_leaf=20,
        random_state=seed
    )

    clf.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    results = {
        'dataset': dataset_name,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'train_pos_rate': train_pos / len(y_train),
        'test_pos_rate': test_pos / len(y_test),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test for spurious correlations in vulnerability datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        help=f'Dataset to test (default: all). Options: {", ".join(DATASET_CONFIGS.keys())}, all'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default='..',
        help='Path to VulnBench root directory (default: ..)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='microsoft/unixcoder-base-nine',
        help='HuggingFace tokenizer to use (default: microsoft/unixcoder-base-nine)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to use (for quick testing). Default: use all'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='spurious_correlations_results.json',
        help='Output file for results (default: spurious_correlations_results.json)'
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Determine which datasets to run
    if args.dataset == 'all':
        datasets_to_run = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_run = [args.dataset]

    print(f"\nWill run experiments on {len(datasets_to_run)} dataset(s)")
    print(f"Max samples per split: {'all' if args.max_samples is None else args.max_samples}")

    # Run experiments
    all_results = []
    for dataset_name in datasets_to_run:
        try:
            result = run_experiment(
                dataset_name,
                args.project_root,
                tokenizer,
                max_samples=args.max_samples,
                seed=args.seed
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR running {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = Path(args.output)
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'F1 Score':<12} {'Accuracy':<12} {'Samples':<10}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['dataset']:<20} {r['f1']:<12.4f} {r['accuracy']:<12.4f} {r['test_samples']:<10}")

    print(f"\nComplete! Results saved to: {output_path}")


if __name__ == '__main__':
    main()
