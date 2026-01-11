#!/usr/bin/env python3
"""
Generate train_all.json from models.json and datasets.json.

This script automatically creates a comprehensive training configuration
that includes all models and datasets, eliminating manual duplication.

Usage:
    python generate_train_all_config.py
    python generate_train_all_config.py --output config/experiments/custom.json
    python generate_train_all_config.py --dataset-group small --seeds 123 456 789
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional


def load_config(config_dir: Path):
    """Load models and datasets configurations."""
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


def _generate_dataset_groups(datasets: dict) -> dict:
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


def generate_train_all_config(
    models_config: dict,
    datasets_config: dict,
    dataset_group: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    epoch: int = 5,
    pos_weight: float = 1.0,
    mode: str = "train",
    use_dataset_groups: bool = False
) -> dict:
    """Generate a comprehensive training configuration.

    Args:
        models_config: Models configuration dict
        datasets_config: Datasets configuration dict
        dataset_group: If specified, only include datasets from this group (e.g., "small", "big", "all")
        seeds: List of seeds to use. Defaults to [123456, 789012, 345678]
        epoch: Number of epochs
        pos_weight: Positive class weight
        mode: "train" or "test"
        use_dataset_groups: If True, use "group:name" instead of expanding to individual datasets
    """
    if seeds is None:
        seeds = [123456, 789012, 345678]

    # Extract all model names
    models = list(models_config["models"].keys())

    # Determine datasets to include
    if dataset_group:
        if dataset_group not in datasets_config.get("dataset_groups", {}):
            raise ValueError(f"Unknown dataset group: {dataset_group}")

        if use_dataset_groups:
            # Use group reference
            datasets = [f"group:{dataset_group}"]
        else:
            # Expand to individual datasets
            datasets = datasets_config["dataset_groups"][dataset_group]
    else:
        if use_dataset_groups:
            # Use group reference for all
            datasets = ["group:all"]
        else:
            # Include all datasets
            datasets = list(datasets_config["datasets"].keys())

    config = {
        "models": models,
        "datasets": datasets,
        "seeds": seeds,
        "pos_weight": pos_weight,
        "epoch": epoch,
        "out_suffix": "",
        "mode": mode,
        "block_size": 400,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "learning_rate": 2e-5,
        "max_grad_norm": 1.0,
        "dropout_probability": 0.2,
        "use_wandb": True,
        "wandb_project": "vulnerability-benchmark",
        "_metadata": {
            "generated_by": "generate_train_all_config.py",
            "total_models": len(models),
            "total_datasets": len(datasets) if not use_dataset_groups else f"group:{dataset_group or 'all'}",
            "total_seeds": len(seeds),
            "total_combinations": len(models) * len(datasets) * len(seeds)
        }
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate train_all.json from models.json and datasets.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default train_all.json with all models and datasets
  python generate_train_all_config.py

  # Generate config for only small datasets
  python generate_train_all_config.py --dataset-group small

  # Generate config with custom seeds
  python generate_train_all_config.py --seeds 111 222 333

  # Generate config using dataset group references (more maintainable)
  python generate_train_all_config.py --use-groups

  # Generate test configuration
  python generate_train_all_config.py --mode test --epoch 1 --seeds 123456
        """
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('config'),
        help='Directory containing models.json and datasets.json (default: config)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('config/experiments/train_all.json'),
        help='Output file path (default: config/experiments/train_all.json)'
    )
    parser.add_argument(
        '--dataset-group',
        choices=['small', 'big', 'all'],
        help='Only include datasets from specified group'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        help='Seeds to use (default: 123456 234567 345678)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=5,
        help='Number of epochs (default: 5)'
    )
    parser.add_argument(
        '--pos-weight',
        type=float,
        default=1.0,
        help='Positive class weight (default: 1.0)'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        default='train',
        help='Training or testing mode (default: train)'
    )
    parser.add_argument(
        '--use-groups',
        action='store_true',
        help='Use "group:name" references instead of expanding to individual datasets'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print config without writing to file'
    )

    args = parser.parse_args()

    # Load configurations
    print(f"Loading configurations from {args.config_dir}")
    models_config, datasets_config = load_config(args.config_dir)

    print(f"Found {len(models_config['models'])} models")
    print(f"Found {len(datasets_config['datasets'])} datasets")
    print(f"Found {len(datasets_config.get('dataset_groups', {}))} dataset groups")

    # Generate config
    config = generate_train_all_config(
        models_config,
        datasets_config,
        dataset_group=args.dataset_group,
        seeds=args.seeds,
        epoch=args.epoch,
        pos_weight=args.pos_weight,
        mode=args.mode,
        use_dataset_groups=args.use_groups
    )

    # Print summary
    print("\n" + "="*80)
    print("GENERATED CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Models ({len(config['models'])}): {', '.join(config['models'])}")

    if args.use_groups:
        print(f"Datasets: {config['datasets']}")
    else:
        print(f"Datasets ({len(config['datasets'])}): {', '.join(config['datasets'])}")

    print(f"Seeds ({len(config['seeds'])}): {config['seeds']}")
    print(f"Epoch: {config['epoch']}")
    print(f"Mode: {config['mode']}")
    print(f"Total experiment combinations: {config['_metadata']['total_combinations']}")

    # Write or print
    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN - Configuration (not written to file):")
        print("="*80)
        print(json.dumps(config, indent=2))
    else:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, 'w') as f:
            json.dump(config, indent=2, fp=f)

        print(f"\n✓ Configuration written to: {args.output}")
        print(f"  Use with: ./run-recipe.sh {args.output}")


if __name__ == "__main__":
    main()
