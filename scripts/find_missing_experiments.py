#!/usr/bin/env python3
"""
Find missing experiments by comparing expected runs (from config) with actual results.

This script:
1. Loads experiment config to determine expected model/dataset/seed combinations
2. Scans results directory to find what was actually run
3. Identifies missing experiments
4. Optionally generates a new config JSON for rerunning missing experiments

Usage:
    python find_missing_experiments.py --config config/experiments/train_all.json
    python find_missing_experiments.py --config config/experiments/train_all.json --generate-config missing_runs.json
"""

import argparse
import json
from pathlib import Path
from typing import Set, Tuple, List, Dict
from collections import defaultdict


def load_config_files(config_dir: Path) -> Tuple[Dict, Dict]:
    """Load models and datasets config files."""
    models_file = config_dir / "models.json"
    datasets_file = config_dir / "datasets.json"

    with open(models_file) as f:
        models_config = json.load(f)

    with open(datasets_file) as f:
        datasets_config = json.load(f)

    # Auto-generate dataset groups from "size" field if not provided
    if "dataset_groups" not in datasets_config:
        datasets_config["dataset_groups"] = _generate_dataset_groups(datasets_config["datasets"])

    return models_config, datasets_config


def _generate_dataset_groups(datasets: Dict) -> Dict:
    """Auto-generate dataset groups based on the 'size' field."""
    groups = {
        "small": [],
        "big": [],
        "all": []
    }

    for name, config in datasets.items():
        groups["all"].append(name)
        size = config.get("size", "small")
        if size == "small":
            groups["small"].append(name)
        elif size == "big":
            groups["big"].append(name)

    # Sort for consistency
    for group in groups.values():
        group.sort()

    return groups


def resolve_dataset_groups(datasets: List[str], datasets_config: Dict) -> List[str]:
    """Resolve dataset group names to actual dataset list."""
    resolved = []

    for ds in datasets:
        if ds.startswith("group:"):
            group_name = ds.replace("group:", "")
            if group_name in datasets_config.get("dataset_groups", {}):
                resolved.extend(datasets_config["dataset_groups"][group_name])
            else:
                print(f"Warning: Unknown dataset group '{group_name}'")
        else:
            resolved.append(ds)

    return list(set(resolved))  # Remove duplicates


def load_experiment_config(config_file: Path, config_dir: Path) -> Dict:
    """Load and parse experiment config file."""
    with open(config_file) as f:
        exp_config = json.load(f)

    # Load models and datasets configs
    models_config, datasets_config = load_config_files(config_dir)

    # Resolve dataset groups
    datasets = resolve_dataset_groups(exp_config["datasets"], datasets_config)

    # Validate models exist in config
    for model in exp_config["models"]:
        if model not in models_config["models"]:
            print(f"Warning: Model '{model}' not found in models.json")

    config = {
        "models": exp_config["models"],
        "datasets": datasets,
        "seeds": exp_config.get("seeds", [123456]),
        "pos_weight": exp_config.get("pos_weight", 1.0),
        "epoch": exp_config.get("epoch", 5),
        "out_suffix": exp_config.get("out_suffix", "splits"),
        "mode": exp_config.get("mode", "train")
    }

    return config


def get_expected_experiments(exp_config: Dict) -> Set[Tuple[str, str, int]]:
    """Get set of expected (model, dataset, seed) tuples."""
    expected = set()

    for model in exp_config["models"]:
        for dataset in exp_config["datasets"]:
            for seed in exp_config["seeds"]:
                expected.add((model, dataset, seed))

    return expected


def get_actual_experiments(results_dir: Path, models_config: Dict, out_suffix: str = "splits") -> Set[Tuple[str, str, int]]:
    """Scan results directory to find actual experiments run.

    Handles both legacy and new directory naming:
    - Legacy: directory named after HuggingFace model (e.g., "codebert-base")
    - New: directory named after config name (e.g., "codebert")

    Args:
        results_dir: Directory containing model subdirectories
        models_config: Models configuration to map legacy names
        out_suffix: Suffix used in experiment names (e.g., "splits")
    """
    actual = set()

    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return actual

    # Create mapping: legacy_dir_name -> config_name
    # e.g., "codebert-base" -> "codebert", "codet5-base" -> "codet5"
    legacy_to_config = {}
    for config_name, config in models_config["models"].items():
        legacy_dir = config["model_name"].split("/")[-1]
        if legacy_dir != config_name:
            legacy_to_config[legacy_dir] = config_name

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        dir_name = model_dir.name

        # Determine model config name
        # First check if it's a legacy directory name
        if dir_name in legacy_to_config:
            model_name = legacy_to_config[dir_name]
        else:
            # Assume it's already the config name
            model_name = dir_name

        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name

            # Parse experiment directory name: "dataset_splits_seed123456"
            if '_seed' in exp_name:
                parts = exp_name.split('_seed')
                dataset_part = parts[0]
                try:
                    seed = int(parts[1])
                except ValueError:
                    continue

                # Remove out_suffix if present (e.g., "cvefixes_splits" -> "cvefixes")
                if out_suffix and dataset_part.endswith(f'_{out_suffix}'):
                    dataset = dataset_part[:-len(f'_{out_suffix}')]
                # Handle pos_weight in name
                elif '_pos' in dataset_part:
                    dataset = dataset_part.split('_pos')[0]
                else:
                    dataset = dataset_part

                # Check if experiment is complete (both checkpoint and threshold results exist)
                checkpoint = exp_dir / "checkpoint-best-acc" / "model.bin"
                threshold_json = exp_dir / "threshold_results.json"
                threshold_txt = exp_dir / "threshold_comparison.txt"

                # Experiment is complete if checkpoint exists AND threshold results exist
                has_checkpoint = checkpoint.exists()
                has_threshold = threshold_json.exists() or threshold_txt.exists()

                if has_checkpoint and has_threshold:
                    actual.add((model_name, dataset, seed))

    return actual


def find_missing_experiments(expected: Set[Tuple], actual: Set[Tuple]) -> List[Tuple]:
    """Find experiments that are expected but not found in results."""
    missing = expected - actual
    return sorted(list(missing), key=lambda x: (x[0], x[1], x[2]))


def generate_missing_config(missing_experiments: List[Tuple],
                            original_config: Dict,
                            output_file: Path):
    """Generate a new experiment config for missing runs."""
    # Group by model and dataset
    missing_by_model_dataset = defaultdict(set)

    for model, dataset, seed in missing_experiments:
        missing_by_model_dataset[(model, dataset)].add(seed)

    # Extract unique models and datasets
    models = sorted(set(exp[0] for exp in missing_experiments))
    datasets = sorted(set(exp[1] for exp in missing_experiments))
    seeds = sorted(set(exp[2] for exp in missing_experiments))

    # Create config
    config = {
        "models": models,
        "datasets": datasets,
        "seeds": seeds,
        "pos_weight": original_config.get("pos_weight", 1.0),
        "epoch": original_config.get("epoch", 5),
        "out_suffix": original_config.get("out_suffix", "splits"),
        "mode": original_config.get("mode", "train"),
        "_metadata": {
            "generated_from": "find_missing_experiments.py",
            "total_missing": len(missing_experiments),
            "missing_details": [
                {"model": m, "dataset": d, "seed": s}
                for m, d, s in missing_experiments
            ]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nGenerated missing runs config: {output_file}")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Seeds: {len(seeds)}")
    print(f"  Total combinations: {len(missing_experiments)}")


def print_missing_summary(missing: List[Tuple], actual: Set[Tuple], expected: Set[Tuple]):
    """Print a summary of missing experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT COVERAGE SUMMARY")
    print("="*80)
    print(f"Expected experiments: {len(expected)}")
    print(f"Completed experiments: {len(actual)}")
    print(f"Missing experiments: {len(missing)}")
    print(f"Completion rate: {100 * len(actual) / len(expected):.1f}%")

    if missing:
        print("\n" + "-"*80)
        print("MISSING EXPERIMENTS BY MODEL")
        print("-"*80)

        by_model = defaultdict(list)
        for model, dataset, seed in missing:
            by_model[model].append((dataset, seed))

        for model in sorted(by_model.keys()):
            exps = by_model[model]
            print(f"\n{model}: {len(exps)} missing")

            # Group by dataset
            by_dataset = defaultdict(list)
            for dataset, seed in exps:
                by_dataset[dataset].append(seed)

            for dataset in sorted(by_dataset.keys()):
                seeds = sorted(by_dataset[dataset])
                print(f"  {dataset:20s}: seeds {seeds}")

        print("\n" + "-"*80)
        print("MISSING EXPERIMENTS BY DATASET")
        print("-"*80)

        by_dataset = defaultdict(list)
        for model, dataset, seed in missing:
            by_dataset[dataset].append((model, seed))

        for dataset in sorted(by_dataset.keys()):
            exps = by_dataset[dataset]
            print(f"\n{dataset}: {len(exps)} missing")

            # Group by model
            by_model = defaultdict(list)
            for model, seed in exps:
                by_model[model].append(seed)

            for model in sorted(by_model.keys()):
                seeds = sorted(by_model[model])
                print(f"  {model:20s}: seeds {seeds}")


def main():
    parser = argparse.ArgumentParser(
        description='Find missing experiments by comparing config with results'
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Experiment config file (e.g., config/experiments/train_all.json)'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('config'),
        help='Directory containing models.json and datasets.json'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('../models'),
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--generate-config',
        type=Path,
        help='Generate a new config file for missing experiments'
    )
    parser.add_argument(
        '--json-output',
        type=Path,
        help='Output missing experiments as JSON'
    )

    args = parser.parse_args()

    # Load experiment config
    print(f"Loading experiment config: {args.config}")
    exp_config = load_experiment_config(args.config, args.config_dir)

    # Load models config for legacy directory mapping
    models_config, _ = load_config_files(args.config_dir)

    # Get expected experiments
    expected = get_expected_experiments(exp_config)
    print(f"Expected {len(expected)} experiment combinations")

    # Scan results
    print(f"\nScanning results directory: {args.results_dir}")
    actual = get_actual_experiments(args.results_dir, models_config, exp_config.get("out_suffix", "splits"))
    print(f"Found {len(actual)} completed experiments")

    # Find missing
    missing = find_missing_experiments(expected, actual)

    # Print summary
    print_missing_summary(missing, actual, expected)

    # Generate config if requested
    if args.generate_config and missing:
        generate_missing_config(missing, exp_config, args.generate_config)

    # Output JSON if requested
    if args.json_output:
        output = {
            "expected": len(expected),
            "completed": len(actual),
            "missing": len(missing),
            "completion_rate": 100 * len(actual) / len(expected) if expected else 0,
            "missing_experiments": [
                {"model": m, "dataset": d, "seed": s}
                for m, d, s in missing
            ]
        }

        with open(args.json_output, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved missing experiments to: {args.json_output}")

    # Exit with appropriate code
    if missing:
        print(f"\n⚠ {len(missing)} experiments still need to be run")
        return 1
    else:
        print("\n✓ All experiments completed!")
        return 0


if __name__ == "__main__":
    exit(main())
