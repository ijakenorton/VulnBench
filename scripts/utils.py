from typing import Dict, Tuple
import os
import sys
import json
from pathlib import Path
import re

def make_dirs_from_string(path_string):
    path = Path(path_string)
    os.makedirs(path.parent, exist_ok=True)

def parse_threshold_json(threshold_json):
    """Parse threshold_results.json file (new format)"""
    try:
        with open(threshold_json, "r") as f:
            data = json.load(f)

        results = {}

        # Extract default threshold metrics
        if "default_threshold" in data and "metrics" in data["default_threshold"]:
            default_metrics = data["default_threshold"]["metrics"]
            results["default_accuracy"] = default_metrics.get("accuracy")
            results["default_precision"] = default_metrics.get("precision")
            results["default_recall"] = default_metrics.get("recall")
            results["default_f1"] = default_metrics.get("f1")

        # Extract optimal threshold metrics
        if "optimal_threshold" in data:
            optimal = data["optimal_threshold"]
            results["optimal_threshold"] = optimal.get("threshold")

            if "metrics" in optimal:
                optimal_metrics = optimal["metrics"]
                results["optimal_accuracy"] = optimal_metrics.get("accuracy")
                results["optimal_precision"] = optimal_metrics.get("precision")
                results["optimal_recall"] = optimal_metrics.get("recall")
                results["optimal_f1"] = optimal_metrics.get("f1")

        # Extract improvement
        if "improvement" in data:
            results["f1_improvement"] = data["improvement"].get("f1")

        # Extract method info
        results["threshold_method"] = data.get("method", "unknown")
        results["optimization_metric"] = data.get("optimization_metric", "unknown")

        # Extract GHOST stats if available
        if "ghost_stats" in data:
            results["ghost_median_score"] = data["ghost_stats"].get(
                "optimal_median_score"
            )
            results["ghost_std_score"] = data["ghost_stats"].get("optimal_std_score")

        return results

    except Exception as e:
        print(f"Error parsing {threshold_json}: {e}")
        return None


def validate_metadata_consistency(
    exp_name, dataset, anonymized, pos_weight, seed, metadata
):
    """
    Validate that metadata is consistent with directory name and data files.
    Returns list of warning messages.
    """
    warnings = []

    # Check anonymized flag consistency with directory name
    has_anon_in_name = "_anon" in exp_name.lower() or "anonymized" in exp_name.lower()
    if anonymized != has_anon_in_name:
        warnings.append(
            f"Anonymized flag mismatch: metadata says {anonymized} but directory name "
            f"{'contains' if has_anon_in_name else 'does not contain'} 'anon'"
        )

    # Check if anonymized flag matches data file paths
    for split in ["train", "valid", "test"]:
        file_key = f"{split}_file"
        if file_key in metadata:
            file_path = metadata[file_key].lower()
            has_anon_in_path = "anon" in file_path or "anonymized" in file_path
            if anonymized != has_anon_in_path:
                warnings.append(
                    f"Anonymized flag mismatch in {split}_file: metadata says {anonymized} "
                    f"but path {'contains' if has_anon_in_path else 'does not contain'} 'anon'"
                )
                break  # Only report once

    # Check dataset name consistency
    if "_seed" in exp_name:
        # Extract dataset from directory name for comparison
        dir_dataset, _, _, _ = parse_legacy_dirname(exp_name)
        if dir_dataset and dir_dataset != dataset:
            warnings.append(
                f"Dataset name mismatch: metadata says '{dataset}' but directory suggests '{dir_dataset}'"
            )

    # Check seed consistency
    seed_in_name = re.search(r"seed(\d+)", exp_name)
    if seed_in_name:
        name_seed = int(seed_in_name.group(1))
        if name_seed != seed:
            warnings.append(
                f"Seed mismatch: metadata says {seed} but directory name has {name_seed}"
            )

    # Validate pos_weight in directory name if present
    if "_pos" in exp_name:
        pos_match = re.search(r"_pos([\d.]+)", exp_name)
        if pos_match:
            name_pos_weight = float(pos_match.group(1))
            if (
                abs(name_pos_weight - pos_weight) > 0.01
            ):  # Allow small floating point differences
                warnings.append(
                    f"pos_weight mismatch: metadata says {pos_weight} but directory name has {name_pos_weight}"
                )

    return warnings


def load_config_files(config_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load models and datasets config files."""
    models_file = config_dir / "models.json"
    datasets_file = config_dir / "datasets.json"
    hardware_file = config_dir / "hardware.json"

    with open(models_file) as f:
        models_config = json.load(f)

    with open(datasets_file) as f:
        datasets_config = json.load(f)

    with open(hardware_file) as f:
        hardware_config = json.load(f)

    # Auto-generate dataset groups from "size" field if not provided
    if "dataset_groups" not in datasets_config:
        datasets_config["dataset_groups"] = _generate_dataset_groups(
            datasets_config["datasets"]
        )

    return models_config, datasets_config, hardware_config

