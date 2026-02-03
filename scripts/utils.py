from typing import Dict, Tuple
import os
import json
from pathlib import Path


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

    # # Auto-generate dataset groups from "size" field if not provided
    # if "dataset_groups" not in datasets_config:
    #     datasets_config["dataset_groups"] = _generate_dataset_groups(
    #         datasets_config["datasets"]
    #     )

    return models_config["models"], datasets_config["datasets"], hardware_config
