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
    # Dataset name may already include the suffix
    if dataset_name.endswith(f"_{out_suffix}"):
        exp_name = f"{dataset_name}_seed{seed}"
    else:
        exp_name = f"{dataset_name}_{out_suffix}_seed{seed}"

    # Try direct model name
    exp_dir = models_dir / model_name / exp_name

    if not exp_dir.exists():
        # Try with -base suffix (legacy naming)
        exp_dir = models_dir / f"{model_name}-base" / exp_name

    if not exp_dir.exists():
        # print(f"    Warning: Directory not found: {exp_dir}")
        return {}

    pred_file = exp_dir / "predictions.txt"
    if not pred_file.exists():
        print(f"    Warning: predictions.txt not found in {exp_dir}")
        return {}

    return load_predictions(pred_file)


def find_dataset_file(data_dir: Path, dataset_name: str) -> Path:
    """Find the dataset file (handles both regular and anonymized versions)."""
    # Remove common suffixes from dataset name
    base_name = dataset_name.replace('_splits', '').replace('_anonymized', '')

    dataset_file = data_dir / base_name / f"{base_name}_full_dataset.jsonl"
    if dataset_file.exists():
        return dataset_file

    anon_file = data_dir / base_name / f"{base_name}_full_dataset_anonymized.jsonl"
    if anon_file.exists():
        return anon_file

    raise FileNotFoundError(f"Dataset file not found for {dataset_name} (tried {base_name})")


def analyze_differential_performance(
    dataset_name: str,
    best_model: str,
    other_models: List[str],
    seed: int,
    models_dir: Path,
    data_dir: Path,
    out_suffix: str = "splits"
) -> Dict:
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
    }

    best_preds = all_predictions[best_model]

    for idx in best_preds.keys():
        if idx not in ground_truth:
            continue

        label = ground_truth[idx]
        best_logit, best_pred = best_preds[idx]
        best_category = categorize_prediction(best_pred, label)

        # Check if best model is correct
        if best_category not in ['TP', 'TN']:
            continue

        # Check other models
        others_wrong = []
        others_right = []
        for model in other_models:
            if model not in all_predictions or idx not in all_predictions[model]:
                continue

            other_logit, other_pred = all_predictions[model][idx]
            other_category = categorize_prediction(other_pred, label)

            if other_category in ['TP', 'TN']:
                others_right.append(model)
            else:
                others_wrong.append(model)

        # Skip if no other models had predictions for this sample
        if not others_wrong and not others_right:
            continue

        # Skip if best model isn't doing better than others
        if len(others_wrong) == 0:
            continue

        # This is a differential example!
        example = {
            'index': idx,
            'label': label,
            'code': code_samples.get(idx, ''),
            'best_model': best_model,
            'best_logit': best_logit,
            'best_pred': best_pred,
            'best_category': best_category,
            'other_predictions': {},
            'num_others_wrong': len(others_wrong),
            'num_others_right': len(others_right)
        }

        for model in other_models:
            if model in all_predictions and idx in all_predictions[model]:
                other_logit, other_pred = all_predictions[model][idx]
                other_category = categorize_prediction(other_pred, label)
                example['other_predictions'][model] = {
                    'logit': other_logit,
                    'pred': other_pred,
                    'category': other_category
                }

        # Categorize the type of differential
        if len(others_right) == 0:
            results['best_correct_all_others_wrong'].append(example)
        if len(others_wrong) > len(others_right):
            results['best_correct_most_others_wrong'].append(example)

        if best_category == 'TP':
            results['best_tp_others_fn'].append(example)
        elif best_category == 'TN':
            results['best_tn_others_fp'].append(example)

    # Print summary
    print(f"\nDifferential Performance Summary:")
    print(f"  Best model right, ALL others wrong: {len(results['best_correct_all_others_wrong'])}")
    print(f"  Best model right, MOST others wrong: {len(results['best_correct_most_others_wrong'])}")
    print(f"  Best found vulnerability, others missed (TP vs FN): {len(results['best_tp_others_fn'])}")
    print(f"  Best correctly safe, others false alarm (TN vs FP): {len(results['best_tn_others_fp'])}")

    return results


def print_examples(results: Dict, max_examples: int = 5):
    """Pretty print example code snippets."""
    for category, examples in results.items():
        if not examples:
            continue

        print(f"\n{'='*80}")
        print(f"{category.upper().replace('_', ' ')}")
        print(f"{'='*80}")

        for i, ex in enumerate(examples[:max_examples]):
            print(f"\n--- Example {i+1}/{min(len(examples), max_examples)} (Index: {ex['index']}) ---")
            print(f"Label: {ex['label']} ({'Vulnerable' if ex['label'] == 1 else 'Safe'})")
            print(f"\nBest Model ({ex['best_model']}):")
            print(f"  Logit: {ex['best_logit']:.4f}, Prediction: {ex['best_pred']} ({ex['best_category']})")

            print(f"\nOther Models:")
            for model, preds in ex['other_predictions'].items():
                print(f"  {model}: Logit={preds['logit']:.4f}, Pred={preds['pred']} ({preds['category']})")

            print(f"\nCode:")
            print("-" * 80)
            code = ex['code'][:1000]  # Limit code length
            if len(ex['code']) > 1000:
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
