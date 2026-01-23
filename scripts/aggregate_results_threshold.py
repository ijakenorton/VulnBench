#!/usr/bin/env python3
"""
Post-processing script to aggregate results across multiple seeds
Usage: python aggregate_results.py --results_dir ../models --output results_summary.csv
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re


def extract_results_from_directory(results_dir):
    """
    Extract results from all experiment directories

    Now uses metadata files (data_split_info.json) instead of parsing directory names.
    This makes the extraction more robust and handles anonymized datasets correctly.

    Expected structure:
    models/
    ├── codebert/
    │   ├── diversevul_seed123_splits/
    │   │   ├── experiment_summary.txt
    │   │   ├── threshold_comparison.txt
    │   │   ├── predictions.txt
    │   │   └── data_split_info.json  # Metadata file
    │   ├── diversevul_seed456_splits/
    │   └── icvul_anon_seed123_splits/  # Anonymized dataset
    ├── natgen/
    └── ...
    """
    all_results = []
    missing_results = []
    validation_warnings = []

    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name

            # Read metadata from data_split_info.json
            metadata_file = exp_dir / "data_split_info.json"

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    # Extract parameters from metadata
                    dataset = metadata.get("dataset_name", "unknown")
                    anonymized = metadata.get("anonymized", False)

                    # If dataset is unknown (old metadata), try to extract from directory name
                    if dataset == "unknown":
                        dataset_from_dir, _, _, anon_from_dir = parse_legacy_dirname(
                            exp_name
                        )
                        if dataset_from_dir:
                            dataset = dataset_from_dir
                            # If anonymized wasn't in metadata, use value from directory
                            if not metadata.get("anonymized"):
                                anonymized = anon_from_dir

                    # Get seed from metadata (prefer from filename for backwards compatibility)
                    # Extract seed from train/val/test file paths or split_files
                    seed = None
                    for split in ["train", "valid", "test"]:
                        # Try new format (split_files dict)
                        split_files = metadata.get("split_files", {})
                        if split in split_files:
                            file_path = split_files[split]
                            seed_match = re.search(r"seed(\d+)", file_path)
                            if seed_match:
                                seed = int(seed_match.group(1))
                                break

                        # Try old format (train_file, valid_file, test_file)
                        file_key = f"{split}_file"
                        if file_key in metadata:
                            file_path = metadata[file_key]
                            seed_match = re.search(r"seed(\d+)", file_path)
                            if seed_match:
                                seed = int(seed_match.group(1))
                                break

                    # Fallback: try to get seed from directory name or metadata directly
                    if seed is None:
                        # Check if seed is in metadata directly
                        seed = metadata.get("seed")
                        if seed is None:
                            seed_match = re.search(r"seed(\d+)", exp_name)
                            if seed_match:
                                seed = int(seed_match.group(1))
                            else:
                                print(
                                    f"  ⚠ {exp_name}: Could not extract seed from metadata or directory name"
                                )
                                seed = 0  # Default to 0 if we can't find it

                    # Get hyperparameters from metadata
                    hyperparams = metadata.get("hyperparameters", {})
                    # Fallback to legacy training_args if hyperparameters not available
                    if not hyperparams:
                        hyperparams = metadata.get("training_args", {})

                    pos_weight = hyperparams.get("pos_weight", 1.0)
                    learning_rate = hyperparams.get("learning_rate", 2e-5)
                    dropout = hyperparams.get("dropout_probability", 0.1)
                    epochs = hyperparams.get("epochs", 5)

                    # Validate metadata consistency with directory name
                    warnings = validate_metadata_consistency(
                        exp_name, dataset, anonymized, pos_weight, seed, metadata
                    )
                    if warnings:
                        validation_warnings.extend(warnings)
                        for warning in warnings:
                            print(f"  ⚠ {exp_name}: {warning}")

                except Exception as e:
                    print(f"  ✗ {exp_name}: Error reading metadata: {e}")
                    # Fallback to legacy directory name parsing
                    dataset, seed, pos_weight, anonymized = parse_legacy_dirname(
                        exp_name
                    )
                    if dataset is None:
                        continue
            else:
                # No metadata file - use legacy directory name parsing
                print(f"  ℹ {exp_name}: No metadata file, using legacy parsing")
                dataset, seed, pos_weight, anonymized = parse_legacy_dirname(exp_name)
                if dataset is None:
                    continue

            # Look for results files - prefer JSON, fallback to txt
            threshold_json = exp_dir / "threshold_results.json"
            threshold_txt = exp_dir / "threshold_comparison.txt"
            summary_file = exp_dir / "experiment_summary.txt"

            if threshold_json.exists():
                results = parse_threshold_json(threshold_json)
            elif threshold_txt.exists():
                results = parse_threshold_comparison(threshold_txt)
            else:
                results = None

            if results:
                results.update(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "seed": seed,
                        "pos_weight": pos_weight,
                        "anonymized": anonymized,
                        "exp_dir": str(exp_dir),
                    }
                )
                all_results.append(results)
                improvement = results.get("f1_improvement", 0)
                anon_flag = " [ANON]" if anonymized else ""
                print(
                    f"  ✓ {exp_name}{anon_flag}: F1={results.get('optimal_f1', 'N/A'):.4f}, Improvement={improvement:+.4f}"
                )
            elif threshold_json.exists() or threshold_txt.exists():
                print(f"  ✗ {exp_name}: Could not parse results")
            else:
                missing = {
                    "dir": str(exp_dir),
                    "name": exp_name,
                    "seed": seed,
                    "dataset": dataset,
                    "model_name": model_name,
                    "anonymized": anonymized,
                }
                missing_results.append(missing)
                print(
                    f"  ✗ {exp_name}: No threshold_results.json or threshold_comparison.txt found"
                )

    # Report validation warnings at the end
    if validation_warnings:
        print("\n" + "=" * 80)
        print(f"VALIDATION WARNINGS ({len(validation_warnings)} total)")
        print("=" * 80)
        for warning in validation_warnings[:10]:  # Show first 10
            print(f"  • {warning}")
        if len(validation_warnings) > 10:
            print(f"  ... and {len(validation_warnings) - 10} more warnings")

    return all_results, missing_results


def parse_legacy_dirname(exp_name):
    """
    Parse directory name using legacy format (for backwards compatibility)
    Returns: (dataset, seed, pos_weight, anonymized)
    """
    # Handle pattern: "dataset_seed123" or "dataset_pos2.0_seed123" or "dataset_anon_seed123_splits"
    if "_seed" not in exp_name:
        return None, None, None, False

    # Check for anonymized flag
    anonymized = "_anon" in exp_name or "anonymized" in exp_name.lower()

    # Extract seed
    seed_match = re.search(r"_seed(\d+)", exp_name)
    if not seed_match:
        return None, None, None, False
    seed = int(seed_match.group(1))

    # Get the part before seed
    parts = exp_name.split("_seed")[0]

    # Remove 'anon' if present
    parts = parts.replace("_anon", "")

    # Check for pos_weight
    pos_weight = 1.0
    if "_pos" in parts:
        pos_parts = parts.split("_pos")
        dataset = pos_parts[0]
        # Extract pos_weight value (may have other suffixes after it)
        pos_value_str = pos_parts[1].split("_")[0]
        try:
            pos_weight = float(pos_value_str)
        except ValueError:
            pass  # Keep default
    else:
        # Remove common suffixes to get clean dataset name
        dataset = parts.replace("_splits", "").replace("_split", "")

    return dataset, seed, pos_weight, anonymized


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


def parse_threshold_comparison(threshold_file):
    """Parse the threshold_comparison.txt file to extract metrics"""
    try:
        with open(threshold_file, "r") as f:
            content = f.read()

        results = {}

        # Extract default threshold results
        default_match = re.search(
            r"Default Threshold \(0\.5\):\s*\n(.*?)(?=\n\nOptimal|$)",
            content,
            re.DOTALL,
        )
        if default_match:
            default_section = default_match.group(1)
            results["default_accuracy"] = extract_metric(default_section, "accuracy")
            results["default_precision"] = extract_metric(default_section, "precision")
            results["default_recall"] = extract_metric(default_section, "recall")
            results["default_f1"] = extract_metric(default_section, "f1")

        # Extract optimal threshold results
        # Handle both old format: "Optimal Threshold (0.38):"
        # and new format: "Optimal Threshold (0.38) [ghost]:"
        optimal_match = re.search(
            r"Optimal Threshold \(([\d.]+)\)(?:\s*\[[\w_]+\])?:\s*\n(.*?)(?=\n\n(?:Grid Search|Improvement)|$)",
            content,
            re.DOTALL,
        )
        if optimal_match:
            optimal_threshold = float(optimal_match.group(1))
            optimal_section = optimal_match.group(2)
            results["optimal_threshold"] = optimal_threshold
            results["optimal_accuracy"] = extract_metric(optimal_section, "accuracy")
            results["optimal_precision"] = extract_metric(optimal_section, "precision")
            results["optimal_recall"] = extract_metric(optimal_section, "recall")
            results["optimal_f1"] = extract_metric(optimal_section, "f1")

        # Extract improvement - handle both old and new formats
        # Old format: "Improvement: F1 +0.0711, Precision +0.1102, Recall +0.0154"
        # New format: "Improvement over default (0.5):\n  F1: +0.0711\n  Precision: ..."
        improvement_match = re.search(
            r"Improvement(?:\s+over\s+default[^:]*)?:\s*F1[:\s]+([+-][\d.]+)",
            content,
            re.IGNORECASE,
        )
        if improvement_match:
            results["f1_improvement"] = float(improvement_match.group(1))

        return results

    except Exception as e:
        print(f"Error parsing {threshold_file}: {e}")
        return None


def extract_metric(text, metric_name):
    """Extract a specific metric value from text"""
    pattern = rf"{metric_name}:\s*([\d.]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def aggregate_results(all_results):
    """Aggregate results by model, dataset, pos_weight, and anonymized flag"""
    df = pd.DataFrame(all_results)

    if df.empty:
        print("No results found!")
        return pd.DataFrame()

    # Ensure anonymized column exists
    if "anonymized" not in df.columns:
        df["anonymized"] = False

    # Group by model, dataset, pos_weight, and anonymized flag
    # CRITICAL: Separate anonymized experiments to prevent mixing results
    grouped = df.groupby(["model", "dataset", "pos_weight", "anonymized"])

    aggregated = []

    for (model, dataset, pos_weight, anonymized), group in grouped:
        if len(group) < 2:
            anon_label = " [ANON]" if anonymized else ""
            print(
                f"Warning: Only {len(group)} run(s) for {model}/{dataset}/pos{pos_weight}{anon_label}"
            )

        # Calculate mean and std for key metrics
        metrics = [
            "optimal_f1",
            "optimal_accuracy",
            "optimal_precision",
            "optimal_recall",
            "default_f1",
            "default_accuracy",
            "f1_improvement",
            "optimal_threshold",
        ]

        agg_result = {
            "model": model,
            "dataset": dataset,
            "pos_weight": pos_weight,
            "anonymized": anonymized,
            "n_seeds": len(group),
            "seeds": sorted(group["seed"].tolist()),
        }

        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    agg_result[f"{metric}_mean"] = values.mean()
                    agg_result[f"{metric}_std"] = (
                        values.std() if len(values) > 1 else 0.0
                    )
                    agg_result[f"{metric}_min"] = values.min()
                    agg_result[f"{metric}_max"] = values.max()

        aggregated.append(agg_result)

    return pd.DataFrame(aggregated)


def format_results_table(df):
    """Format results for publication"""
    if df.empty:
        return df

    # Create publication-ready table
    pub_df = df.copy()

    # Format mean ± std columns
    for metric in [
        "optimal_f1",
        "optimal_accuracy",
        "optimal_precision",
        "optimal_recall",
    ]:
        if f"{metric}_mean" in df.columns and f"{metric}_std" in df.columns:
            pub_df[f"{metric}_formatted"] = pub_df.apply(
                lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}",
                axis=1,
            )

    # Add anonymized indicator to dataset name
    if "anonymized" in pub_df.columns:
        pub_df["dataset_display"] = pub_df.apply(
            lambda row: (
                f"{row['dataset']} [ANON]" if row["anonymized"] else row["dataset"]
            ),
            axis=1,
        )
    else:
        pub_df["dataset_display"] = pub_df["dataset"]

    # Select and reorder columns for publication
    pub_columns = [
        "model",
        "dataset_display",
        "pos_weight",
        "n_seeds",
        "optimal_f1_formatted",
        "optimal_accuracy_formatted",
        "optimal_precision_formatted",
        "optimal_recall_formatted",
    ]

    pub_df = pub_df[pub_columns].copy()
    pub_df.columns = [
        "Model",
        "Dataset",
        "Pos Weight",
        "Seeds",
        "F1",
        "Accuracy",
        "Precision",
        "Recall",
    ]

    return pub_df


def format_improvement_table(df):
    """Format threshold optimization improvement analysis table"""
    if df.empty:
        return df

    # Create improvement analysis table
    imp_df = df.copy()

    # Format improvement columns
    if "f1_improvement_mean" in df.columns and "f1_improvement_std" in df.columns:
        imp_df["f1_improvement_formatted"] = imp_df.apply(
            lambda row: f"{row['f1_improvement_mean']:+.3f} ± {row['f1_improvement_std']:.3f}",
            axis=1,
        )

    if "optimal_threshold_mean" in df.columns and "optimal_threshold_std" in df.columns:
        imp_df["optimal_threshold_formatted"] = imp_df.apply(
            lambda row: f"{row['optimal_threshold_mean']:.3f} ± {row['optimal_threshold_std']:.3f}",
            axis=1,
        )

    # Calculate percentage improvement (handle division by zero)
    if "default_f1_mean" in df.columns and "f1_improvement_mean" in df.columns:

        def calc_percent_improvement(row):
            if row["default_f1_mean"] < 0.001:  # Nearly zero
                if row["f1_improvement_mean"] > 0.001:
                    return float("inf")  # Improved from nothing
                else:
                    return 0.0  # Still nothing
            return (row["f1_improvement_mean"] / row["default_f1_mean"]) * 100

        imp_df["percent_improvement"] = imp_df.apply(calc_percent_improvement, axis=1)

        def format_percent(x):
            if x == float("inf"):
                return "+∞ (baseline=0)"
            elif x == float("-inf"):
                return "-∞"
            elif pd.isna(x):
                return "N/A"
            else:
                return f"{x:+.1f}%"

        imp_df["percent_improvement_formatted"] = imp_df["percent_improvement"].apply(
            format_percent
        )

    # Format default F1 for reference
    if "default_f1_mean" in df.columns and "default_f1_std" in df.columns:
        imp_df["default_f1_formatted"] = imp_df.apply(
            lambda row: f"{row['default_f1_mean']:.3f} ± {row['default_f1_std']:.3f}",
            axis=1,
        )

    # Add anonymized indicator to dataset name
    if "anonymized" in imp_df.columns:
        imp_df["dataset_display"] = imp_df.apply(
            lambda row: (
                f"{row['dataset']} [ANON]" if row["anonymized"] else row["dataset"]
            ),
            axis=1,
        )
    else:
        imp_df["dataset_display"] = imp_df["dataset"]

    # Select columns for improvement table
    imp_columns = [
        "model",
        "dataset_display",
        "n_seeds",
        "default_f1_formatted",
        "optimal_threshold_formatted",
        "f1_improvement_formatted",
        "percent_improvement_formatted",
    ]

    imp_df = imp_df[imp_columns].copy()
    imp_df.columns = [
        "Model",
        "Dataset",
        "Seeds",
        "Default F1 (0.5)",
        "Optimal Threshold",
        "F1 Improvement",
        "Relative Gain",
    ]

    return imp_df


def report_stats(aggregated_df, args):

    print(f"Aggregated into {len(aggregated_df)} model/dataset combinations")

    # Save detailed results
    detailed_output = args.output.replace(".csv", "_detailed.csv")
    aggregated_df.to_csv(detailed_output, index=False)
    print(f"Saved detailed results to: {detailed_output}")

    # Create publication table
    pub_df = format_results_table(aggregated_df)
    pub_df.to_csv(args.output, index=False)
    print(f"Saved publication table to: {args.output}")

    # Create improvement analysis table
    improvement_output = args.output.replace(".csv", "_improvements.csv")
    imp_df = format_improvement_table(aggregated_df)
    imp_df.to_csv(improvement_output, index=False)
    print(f"Saved improvement analysis to: {improvement_output}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(pub_df.to_string(index=False))

    # Print improvement analysis
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print(imp_df.to_string(index=False))

    # Identify best performing model per dataset
    print("\n" + "=" * 80)
    print("BEST MODELS PER DATASET")
    print("=" * 80)

    for dataset in aggregated_df["dataset"].unique():
        dataset_results = aggregated_df[aggregated_df["dataset"] == dataset]
        best_idx = dataset_results["optimal_f1_mean"].idxmax()
        best = dataset_results.loc[best_idx]

        print(
            f"{dataset:15s}: {best['model']:12s} "
            f"F1={best['optimal_f1_mean']:.3f}±{best['optimal_f1_std']:.3f} "
            f"({best['n_seeds']} seeds)"
        )

    # Summary statistics for improvements
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY STATISTICS")
    print("=" * 80)

    if "f1_improvement_mean" in aggregated_df.columns:
        improvements = aggregated_df["f1_improvement_mean"].dropna()
        if len(improvements) > 0:
            print(f"Average F1 improvement: {improvements.mean():+.3f}")
            print(f"Median F1 improvement:  {improvements.median():+.3f}")
            print(f"Best improvement:       {improvements.max():+.3f}")
            print(f"Worst improvement:      {improvements.min():+.3f}")

            # Count positive improvements
            positive_improvements = (improvements > 0).sum()
            total_experiments = len(improvements)
            print(
                f"Positive improvements:  {positive_improvements}/{total_experiments} ({100*positive_improvements/total_experiments:.1f}%)"
            )

            # Calculate percentage improvements if available
            if "percent_improvement" in aggregated_df.columns:
                pct_improvements = aggregated_df["percent_improvement"].dropna()
                if len(pct_improvements) > 0:
                    print(f"Average % improvement:  {pct_improvements.mean():+.1f}%")
                    print(f"Median % improvement:   {pct_improvements.median():+.1f}%")
                    print(f"Best % improvement:     {pct_improvements.max():+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results across seeds"
    )
    parser.add_argument(
        "--results_dir", default="../models", help="Directory containing model results"
    )
    parser.add_argument(
        "--output", default="results_summary.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--min_seeds", default=3, type=int, help="Minimum seeds required for inclusion"
    )
    parser.add_argument(
        "--json", action="store_true", help="Also output results as JSON"
    )
    parser.add_argument(
        "--missing-output",
        default="missing_results.json",
        help="Output JSON file for missing results",
    )

    args = parser.parse_args()

    print(f"Scanning results directory: {args.results_dir}")
    all_results, missing_results = extract_results_from_directory(args.results_dir)

    if not all_results:
        print("No results found!")
        return

    print(f"\nFound {len(all_results)} individual experiment results")

    # Aggregate results
    aggregated_df_full = aggregate_results(all_results)
    # Filter by minimum seeds
    aggregated_df = aggregated_df_full[aggregated_df_full["n_seeds"] >= args.min_seeds]
    report_stats(aggregated_df, args)

    # Save missing results to JSON
    with open(args.missing_output, "w") as f:
        json.dump(
            {"missing": missing_results, "count": len(missing_results)}, f, indent=2
        )
    print(f"\nSaved missing results to: {args.missing_output}")

    # Output results as JSON if requested
    if args.json:
        json_output = args.output.replace(".csv", ".json")

        # Convert DataFrame to JSON-friendly format
        results_json = {
            "summary": aggregated_df.to_dict("records"),
            "metadata": {
                "total_experiments": len(all_results),
                "aggregated_combinations": len(aggregated_df),
                "missing_count": len(missing_results),
                "min_seeds_filter": args.min_seeds,
            },
        }

        with open(json_output, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved JSON results to: {json_output}")

    aggregated_df = aggregated_df_full[aggregated_df_full["n_seeds"] != 3]
    if aggregated_df.size != 0:
        print("*" * 80)
        print("*" * 80)
        print("Limited Results")
        report_stats(aggregated_df, args)


if __name__ == "__main__":
    main()
