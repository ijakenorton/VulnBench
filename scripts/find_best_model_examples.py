#!/usr/bin/env python3
"""
Find specific code examples where the best model succeeds and others fail.

This script:
1. Identifies the best performing model per dataset
2. Loads predictions from all models
3. Finds examples where:
   - Best model correctly predicts (TP or TN)
   - Other models incorrectly predict (FP or FN)
4. Outputs the actual code snippets for manual inspection
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from schemas import ConfigLoader


def load_aggregate_results(results_file: Path) -> pd.DataFrame:
    """Load aggregated results to identify best models."""
    with open(results_file) as f:
        data = json.load(f)

    # Handle both direct list and nested structure
    if isinstance(data, dict) and 'summary' in data:
        records = data['summary']
    else:
        records = data

    return pd.DataFrame(records)


def get_best_model_per_dataset(df: pd.DataFrame) -> Dict[str, str]:
    """Identify the best model for each dataset based on F1 score."""
    best_models = {}

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        # Use optimal_f1_mean if available, otherwise optimal_f1
        f1_col = 'optimal_f1_mean' if 'optimal_f1_mean' in dataset_df.columns else 'optimal_f1'
        if f1_col not in dataset_df.columns:
            print(f"Warning: No F1 column found for {dataset}")
            continue

        # Skip datasets with all NaN F1 values
        if dataset_df[f1_col].isna().all():
            print(f"Warning: All F1 values are NaN for {dataset}, skipping...")
            continue

        best_idx = dataset_df[f1_col].idxmax()
        best_model = dataset_df.loc[best_idx, 'model']
        best_f1 = dataset_df.loc[best_idx, f1_col]

        best_models[dataset] = best_model
        print(f"Best model for {dataset}: {best_model} (F1={best_f1:.4f})")

    return best_models


def load_predictions(pred_file: Path) -> Dict[int, Tuple[float, int]]:
    """
    Load predictions from a predictions.txt file.

    Returns:
        Dict mapping index -> (logit, prediction)
    """
    predictions = {}

    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # Format: index\tlogit\tprediction
                idx = int(parts[0])
                logit = float(parts[1])
                pred = int(parts[2])
                predictions[idx] = (logit, pred)
            elif len(parts) == 2:
                # Format: index\tprediction (no logit)
                idx = int(parts[0])
                pred = int(parts[1])
                predictions[idx] = (float(pred), pred)  # Use pred as logit placeholder

    return predictions


def load_ground_truth(dataset_file: Path) -> Dict[int, int]:
    """Load ground truth labels from dataset file."""
    labels = {}

    with open(dataset_file) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            labels[idx] = data['target']

    return labels


def load_code_samples(dataset_file: Path) -> Dict[int, str]:
    """Load actual code samples."""
    samples = {}

    with open(dataset_file) as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            samples[idx] = data.get('func', data.get('code', ''))

    return samples


def categorize_prediction(pred: int, label: int) -> str:
    """Categorize a single prediction."""
    if pred == 1 and label == 1:
        return 'TP'
    elif pred == 0 and label == 0:
        return 'TN'
    elif pred == 1 and label == 0:
        return 'FP'
    else:  # pred == 0 and label == 1
        return 'FN'


def find_model_predictions(
    models_dir: Path,
    model_name: str,
    dataset_name: str,
    seed: int,
    out_suffix: str = "splits"
) -> Dict[int, Tuple[float, int]]:
    """Find and load predictions for a specific model/dataset/seed."""
    # Strip common suffixes from dataset name to get base name
    base_dataset = dataset_name.replace("_splits", "").replace("_split", "")

    # Try multiple naming patterns that match _generate_experiment_canonical_name
    patterns_to_try = [
        f"{base_dataset}_seed{seed}",                    # Standard: devign_seed123456
        f"{base_dataset}_seed{seed}_{out_suffix}",       # With suffix: devign_seed123456_splits
        f"{dataset_name}_seed{seed}",                    # Original name with seed
    ]

    # Model directory names to try
    model_dirs_to_try = [
        model_name,           # e.g., codebert
        f"{model_name}-base", # e.g., codebert-base (legacy)
    ]

    for model_dir in model_dirs_to_try:
        model_path = models_dir / model_dir
        if not model_path.exists():
            continue

        for pattern in patterns_to_try:
            exp_dir = model_path / pattern
            if exp_dir.exists():
                pred_file = exp_dir / "predictions.txt"
                if pred_file.exists():
                    return load_predictions(pred_file)

    # If not found with exact patterns, try glob matching
    for model_dir in model_dirs_to_try:
        model_path = models_dir / model_dir
        if not model_path.exists():
            continue

        # Match directories starting with dataset name and containing seed
        for exp_dir in model_path.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(base_dataset) and f"seed{seed}" in exp_dir.name:
                # Skip anonymized versions for now
                if "_anon" in exp_dir.name:
                    continue
                pred_file = exp_dir / "predictions.txt"
                if pred_file.exists():
                    return load_predictions(pred_file)

    return {}


def find_dataset_file(data_dir: Path, dataset_name: str) -> Path | None:
    """Find the dataset file (handles both regular and anonymized versions)."""
    # Remove common suffixes from dataset name
    base_name = dataset_name.replace('_splits', '').replace('_anonymized', '')
    dataset_file = data_dir / base_name / f"{base_name}_full_dataset.jsonl"
    if dataset_file.exists():
        return dataset_file

    anon_file = data_dir / base_name / f"{base_name}_full_dataset_anonymized.jsonl"
    if anon_file.exists():
        return anon_file

    return None

def analyze_differential_performance(
    dataset_name: str,
    best_model: str,
    other_models: List[str],
    seed: int,
    models_dir: Path,
    data_dir: Path,
    out_suffix: str = "splits"
) -> Dict | None:
    """
    Find examples where best model succeeds and others fail.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset_name}: Best={best_model} vs Others={other_models}")
    print(f"{'='*80}")

    # Load ground truth and code samples
    dataset_file = find_dataset_file(data_dir, dataset_name)
    ground_truth = load_ground_truth(dataset_file)
    code_samples = load_code_samples(dataset_file)

    print(f"Loaded {len(ground_truth)} samples from {dataset_file.name}")

    # Load predictions for all models
    all_predictions = {}
    for model in [best_model] + other_models:
        preds = find_model_predictions(models_dir, model, dataset_name, seed, out_suffix)
        if preds:
            all_predictions[model] = preds
            print(f"  {model}: {len(preds)} predictions")

    if best_model not in all_predictions:
        print(f"Warning: No predictions for best model {best_model}")
        return {}

    # Filter other_models to only those with predictions
    other_models = [m for m in other_models if m in all_predictions]
    print(f"\nComparing best model '{best_model}' against {len(other_models)} other models with predictions")

    if not other_models:
        print(f"Warning: No other models have predictions for {dataset_name}")
        return {}

    # Find differential examples
    results = {
        'best_correct_all_others_wrong': [],  # Best model right, ALL others wrong
        'best_correct_most_others_wrong': [],  # Best model right, majority of others wrong
        'best_tp_others_fn': [],  # Best found vulnerability, others missed
        'best_tn_others_fp': [],  # Best correctly safe, others false alarm
        'all_models_wrong': [],  # ALL models got it wrong (hardest examples)
        'all_models_wrong_fn': [],  # All models missed a vulnerability (FN)
        'all_models_wrong_fp': [],  # All models had false alarm (FP)
    }

    # Get all indices that have predictions from any model
    all_indices = set()
    for model, preds in all_predictions.items():
        all_indices.update(preds.keys())

    for idx in all_indices:
        if idx not in ground_truth:
            continue

        label = ground_truth[idx]
        code = code_samples.get(idx, '')

        # Collect predictions from all models for this example
        model_results = {}
        for model, preds in all_predictions.items():
            if idx in preds:
                logit, pred = preds[idx]
                category = categorize_prediction(pred, label)
                model_results[model] = {
                    'logit': logit,
                    'pred': pred,
                    'category': category
                }

        if not model_results:
            continue

        # Check if ALL models got it wrong
        all_wrong = all(r['category'] in ['FP', 'FN'] for r in model_results.values())
        all_right = all(r['category'] in ['TP', 'TN'] for r in model_results.values())

        if all_wrong:
            example = {
                'index': idx,
                'label': label,
                'code': code,
                'code_length': len(code),
                'predictions': model_results,
                'num_models': len(model_results),
            }
            results['all_models_wrong'].append(example)

            # Subcategorize by error type
            first_category = list(model_results.values())[0]['category']
            if first_category == 'FN':
                results['all_models_wrong_fn'].append(example)
            elif first_category == 'FP':
                results['all_models_wrong_fp'].append(example)

        # Also track differential performance (original logic)
        if best_model in model_results:
            best_result = model_results[best_model]
            best_category = best_result['category']

            if best_category in ['TP', 'TN']:
                others_wrong = [m for m, r in model_results.items()
                               if m != best_model and r['category'] in ['FP', 'FN']]
                others_right = [m for m, r in model_results.items()
                               if m != best_model and r['category'] in ['TP', 'TN']]

                if others_wrong:
                    example = {
                        'index': idx,
                        'label': label,
                        'code': code,
                        'code_length': len(code),
                        'best_model': best_model,
                        'best_logit': best_result['logit'],
                        'best_pred': best_result['pred'],
                        'best_category': best_category,
                        'other_predictions': {m: r for m, r in model_results.items() if m != best_model},
                        'num_others_wrong': len(others_wrong),
                        'num_others_right': len(others_right)
                    }

                    if len(others_right) == 0:
                        results['best_correct_all_others_wrong'].append(example)
                    if len(others_wrong) > len(others_right):
                        results['best_correct_most_others_wrong'].append(example)

                    if best_category == 'TP':
                        results['best_tp_others_fn'].append(example)
                    elif best_category == 'TN':
                        results['best_tn_others_fp'].append(example)

    # Sort "all models wrong" by code length to find simplest examples
    for key in ['all_models_wrong', 'all_models_wrong_fn', 'all_models_wrong_fp']:
        results[key].sort(key=lambda x: x['code_length'])

    # Print summary
    print(f"\nDifferential Performance Summary:")
    print(f"  Best model right, ALL others wrong: {len(results['best_correct_all_others_wrong'])}")
    print(f"  Best model right, MOST others wrong: {len(results['best_correct_most_others_wrong'])}")
    print(f"  Best found vulnerability, others missed (TP vs FN): {len(results['best_tp_others_fn'])}")
    print(f"  Best correctly safe, others false alarm (TN vs FP): {len(results['best_tn_others_fp'])}")

    return results


def print_examples(results: Dict, max_examples: int = 5, categories: List[str] = None):
    """Pretty print example code snippets."""
    for category, examples in results.items():
        if not examples:
            continue

        print(f"\n{'='*80}")
        print(f"{category.upper().replace('_', ' ')}")
        print(f"{'='*80}")

        for i, ex in enumerate(examples[:max_examples]):
            code_len = ex.get('code_length', len(ex.get('code', '')))
            print(f"\n--- Example {i+1}/{min(len(examples), max_examples)} (Index: {ex['index']}, {code_len} chars) ---")
            print(f"Label: {ex['label']} ({'Vulnerable' if ex['label'] == 1 else 'Safe'})")

            # Handle "all models wrong" format
            if 'predictions' in ex:
                print(f"\nAll Model Predictions:")
                for model, preds in ex['predictions'].items():
                    print(f"  {model}: Logit={preds['logit']:.4f}, Pred={preds['pred']} ({preds['category']})")
            else:
                # Original format with best model vs others
                print(f"\nBest Model ({ex['best_model']}):")
                print(f"  Logit: {ex['best_logit']:.4f}, Prediction: {ex['best_pred']} ({ex['best_category']})")

                print(f"\nOther Models:")
                for model, preds in ex['other_predictions'].items():
                    print(f"  {model}: Logit={preds['logit']:.4f}, Pred={preds['pred']} ({preds['category']})")

            print(f"\nCode:")
            print("-" * 80)
            code = ex['code'][:2000]  # Limit code length
            if len(ex['code']) > 2000:
                code += "\n... (truncated)"
            print(code)
            print("-" * 80)

        if len(examples) > max_examples:
            print(f"\n... and {len(examples) - max_examples} more examples")


def main():
    parser = argparse.ArgumentParser(
        description="Find examples where best model succeeds and others fail",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all datasets, using latest results
  python find_best_model_examples.py --results complete_results.json --seed 123456

  # Analyze specific dataset
  python find_best_model_examples.py --results complete_results.json --dataset juliet --seed 123456

  # Show more examples per category
  python find_best_model_examples.py --results complete_results.json --seed 123456 --max-examples 10

  # Save results to JSON
  python find_best_model_examples.py --results complete_results.json --seed 123456 --output examples.json
        """
    )

    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Aggregated results JSON file (e.g., complete_results.json)'
    )
    parser.add_argument(
        '--dataset',
        help='Analyze only this dataset (otherwise analyze all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123456,
        help='Random seed to analyze (default: 123456)'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=5,
        help='Maximum examples to display per category (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'models',
        help='Directory containing model results'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--out-suffix',
        default='splits',
        help='Output suffix for experiment directories'
    )

    args = parser.parse_args()

    # Load aggregated results
    print(f"Loading results from {args.results}...")
    df = load_aggregate_results(args.results)

    # Identify best models
    print(f"\nIdentifying best models per dataset...")
    best_models = get_best_model_per_dataset(df)

    if args.dataset:
        if args.dataset not in best_models:
            print(f"Error: Dataset {args.dataset} not found in results")
            return 1
        datasets = [args.dataset]
    else:
        datasets = list(best_models.keys())

    # Get all models from the results
    all_models = df['model'].unique().tolist()

    # Analyze each dataset
    all_results = {}

    for dataset in datasets:
        best_model = best_models[dataset]
        other_models = [m for m in all_models if m != best_model]

        results = analyze_differential_performance(
            dataset,
            best_model,
            other_models,
            args.seed,
            args.models_dir,
            args.data_dir,
            args.out_suffix
        )

        all_results[dataset] = results

        # Print examples for this dataset
        if args.show_hardest:
            # Only show "all models wrong" categories
            print_examples(results, args.max_examples,
                          categories=['all_models_wrong', 'all_models_wrong_fn', 'all_models_wrong_fp'])
        else:
            print_examples(results, args.max_examples)

    # Save to file if requested
    if args.output:
        # Convert to serializable format
        output_data = {
            'metadata': {
                'seed': args.seed,
                'best_models': best_models,
                'all_models': all_models,
            },
            'results': all_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
