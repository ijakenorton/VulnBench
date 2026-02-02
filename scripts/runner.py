#!/usr/bin/env python3
"""
Main orchestrator for vulnerability detection experiments.

This script replaces the bash-based recipe system with a Python-based
configuration system that's easier to maintain and extend.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple
from datetime import datetime

from schemas import ConfigLoader, ExperimentConfig, ModelConfig, DatasetConfig
from find_missing_experiments import (
    get_expected_experiments,
    get_actual_experiments,
    find_missing_experiments
)


class ExperimentRunner:
    """Runs vulnerability detection experiments."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "scripts"
        self.config_dir = self.scripts_dir / "config"
        self.output_dir = self.project_root / "output"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.code_dir = self.project_root / "Defect-detection" / "code"
        self.history_file = self.scripts_dir / "run_history.jsonl"

        self.config_loader = ConfigLoader(self.config_dir)

    def _setup_env(self, model: ModelConfig, dataset: DatasetConfig,
                   experiment: ExperimentConfig, seed: int) -> dict:
        """Setup environment variables for a run."""
        env = os.environ.copy()

        # Path setup
        env["PROJECT_ROOT"] = str(self.project_root)
        env["SCRIPTS_DIR"] = str(self.scripts_dir)
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["DATA_DIR"] = str(self.data_dir)
        env["MODELS_DIR"] = str(self.models_dir)
        env["LOGS_DIR"] = str(self.logs_dir)
        env["CODE_DIR"] = str(self.code_dir)

        # Model config
        env["model_name"] = model.model_name
        env["tokenizer_name"] = model.tokenizer_name
        env["model_type"] = model.model_type

        # Dataset config
        env["dataset_name"] = dataset.name

        # Experiment config
        env["seed"] = str(seed)
        env["pos_weight"] = str(experiment.pos_weight)
        env["epoch"] = str(experiment.epoch)
        env["out_suffix"] = experiment.out_suffix

        return env

    def _generate_experiment_canonical_name(self, dataset: DatasetConfig, experiment: ExperimentConfig,
                                     seed: int, anonymized: bool = False) -> str:
        """
        Generate experiment directory name with automated hyperparameter tracking.

        Format: {dataset}_{seed}_{suffix}_{hyperparam_changes}
        Example: devign_seed123456_splits_pos2.0_lr5e-5
        """
        parts = []

        # 1. Dataset name
        parts.append(dataset.name)

        # 2. Anonymized flag (CRITICAL for preventing mistakes)
        if anonymized:
            parts.append('anon')

        # 3. Seed
        parts.append(f"seed{seed}")

        # 4. Output suffix (e.g., "splits")
        if experiment.out_suffix:
            parts.append(experiment.out_suffix)

        # 5. Auto-detect non-default hyperparameters
        # Define standard defaults for comparison

        ARGS_DEFAULTS = {
            "pos_weight": 1.0,
            "epoch": 5,
            "learning_rate": 2e-05,
            "max_grad_norm": 1.0,
            "dropout_probability": 0.2,
            "loss_type": "bce",
        }

        ARGS_TRANSFORM = {
            "pos_weight": "pos",
            "epoch": "ep",
            "learning_rate":"lr",
            "max_grad_norm":"gnorm",
            "dropout_probability":"drop",
            "loss_type": "lt",
        }

        hyperparam_suffixes = []

        for key in ARGS_DEFAULTS.keys():
            if hasattr(experiment, key) and experiment[key] != ARGS_DEFAULTS[key]:
                param = f"{ARGS_TRANSFORM[key]}_{experiment[key]}"
                if key == "learning_rate":
                        param = f"{ARGS_TRANSFORM[key]}{experiment[key]:.0e}"

                hyperparam_suffixes.append(param)

        # Add hyperparameter suffixes
        parts.extend(hyperparam_suffixes)

        return '_'.join(parts)

    def _build_python_command(self, model: ModelConfig, dataset: DatasetConfig,
                             experiment: ExperimentConfig, seed: int, model_config_name: str, anonymized: bool = False) -> List[str]:
        """Build the Python command to run."""
        # Use model_config_name (e.g., "codet5" or "codet5-full") to avoid collisions
        # when multiple configs share the same HuggingFace model

        # Generate directory name with automated hyperparameter tracking
        dir_name = self._generate_experiment_canonical_name(dataset, experiment, seed, anonymized)
        output_dir = self.models_dir / model_config_name / dir_name

        # For cross-dataset testing, use source model directory if specified
        if experiment.source_model_dir:
            # Replace {seed} placeholder in source_model_dir pattern
            source_dir_pattern = experiment.source_model_dir.replace("{seed}", str(seed))
            source_model_path = self.models_dir / model_config_name / source_dir_pattern

            # Override output_dir to point to source model for checkpoint loading
            # But we'll use a special flag to save results to the correct location
            checkpoint_dir = source_model_path
        else:
            checkpoint_dir = output_dir

        cmd = [
            "python", str(self.code_dir / "run.py"),
            f"--output_dir={output_dir}",
            f"--model_type={model.model_type}",
            f"--tokenizer_name={model.tokenizer_name}",
            f"--model_name_or_path={model.model_name}",
        ]

        # Add source checkpoint directory if doing cross-dataset testing
        if experiment.source_model_dir:
            cmd.append(f"--source_checkpoint_dir={checkpoint_dir}")

        # Add mode flags
        if experiment.mode == "train":
            cmd.extend(["--do_train", "--do_eval", "--do_test"])
        else:
            cmd.append("--do_test")

        dataset_suffix = "full_dataset.jsonl"

        if anonymized:
            dataset_suffix = "full_dataset_anonymized.jsonl"

        # Add dataset - either use original splits or generate from one_data_file
        if experiment.use_original_splits:
            # Use original train/val/test splits directly
            train_file = self.data_dir / dataset.name / f"{dataset.name}_train.jsonl"
            val_file = self.data_dir / dataset.name / f"{dataset.name}_val.jsonl"
            test_file = self.data_dir / dataset.name / f"{dataset.name}_test.jsonl"

            cmd.extend([
                f"--train_data_file={train_file}",
                f"--eval_data_file={val_file}",
                f"--test_data_file={test_file}",
            ])
        else:
            # Use one_data_file format (will be split by seed)
            data_file = self.data_dir / dataset.name / f"{dataset.name}_{dataset_suffix}"
            cmd.append(f"--one_data_file={data_file}")

        # Add hyperparameters
        cmd.extend([
            f"--epoch={experiment.epoch}",
            f"--block_size={experiment.block_size}",
            f"--train_batch_size={experiment.train_batch_size}",
            f"--eval_batch_size={experiment.eval_batch_size}",
            f"--learning_rate={experiment.learning_rate}",
            f"--max_grad_norm={experiment.max_grad_norm}",
            f"--pos_weight={experiment.pos_weight}",
            f"--dropout_probability={experiment.dropout_probability}",
            f"--seed={seed}",
        ])

        # Add anonymized flag if applicable
        if anonymized:
            cmd.append("--anonymized")

        # Add loss function parameters
        cmd.extend([
            f"--loss_type={experiment.loss_type}",
            f"--cb_beta={experiment.cb_beta}",
            f"--focal_gamma={experiment.focal_gamma}",
        ])

        # Add threshold optimization parameters (inference time)
        cmd.extend([
            f"--threshold_method={experiment.threshold_method}",
            f"--threshold_metric={experiment.threshold_metric}",
            f"--min_recall={experiment.min_recall}",
            f"--threshold_precision_weight={experiment.threshold_precision_weight}",
            f"--ghost_n_subsets={experiment.ghost_n_subsets}",
            f"--ghost_subset_size={experiment.ghost_subset_size}",
        ])

        # Evaluation
        if experiment.mode == "train":
            cmd.append("--evaluate_during_training")

        # Wandb
        if experiment.use_wandb:

            canonical_name = self._generate_experiment_canonical_name(dataset, experiment, seed, anonymized)
            wandb_run_name = f"{model.model_type}_{canonical_name}"
            cmd.extend([
                "--use_wandb",
                f"--wandb_project={experiment.wandb_project}",
                f"--wandb_run_name={wandb_run_name}",
            ])

        return cmd

    def _build_sbatch_command(self, dataset: DatasetConfig, model_name: str,
                             experiment: ExperimentConfig, seed: int,
                             legacy_mode: bool = False, model: ModelConfig = None, anonymized: bool = False) -> List[str]:
        """Build the sbatch command."""

        canonical_name = self._generate_experiment_canonical_name(dataset, experiment, seed, anonymized)
        output_file = self.logs_dir / f"{dataset.name}_out" / f"{canonical_name}_%j.out"

        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True)

        # Gradient boosting uses CPU-only with high memory
        if model and model.model_type == "gradient_boosting":
            sbatch_args = [
                "sbatch",
                "--cpus-per-task=16",
                "--partition=aoraki_bigmem",
                "--mem=256gb",
                f"--job-name={canonical_name}",
                f"--time={dataset.time_hours}:00:00",
                f"--output={output_file}",
            ]
        else:
            sbatch_args = [
                "sbatch",
                "--gpus-per-node=1",
                f"--partition={dataset.gpu}",
                "--mem=128gb",
                f"--job-name={canonical_name}",
                f"--time={dataset.time_hours}:00:00",
                f"--output={output_file}",
            ]

        return sbatch_args

    def _build_sbatch_wrap_command(self, model: ModelConfig, dataset: DatasetConfig,
                                   experiment: ExperimentConfig, seed: int, model_config_name: str, anonymized: False = False) -> str:
        """Build a complete sbatch --wrap command for direct Python execution."""
        python_cmd = self._build_python_command(model, dataset, experiment, seed, model_config_name, anonymized)

        # Build the full command with conda activation
        conda_activate = "source ~/miniconda3/etc/profile.d/conda.sh"
        conda_env = "conda activate ensemble"
        python_exec = " ".join(python_cmd)

        # Combine into a single shell command
        wrap_cmd = f"{conda_activate} && {conda_env} && {python_exec}"

        return wrap_cmd

    def _log_run_history(self, experiment_file: Path, experiment: ExperimentConfig,
                        job_ids: List[str], runs: List[Tuple[str, str, int]],
                        mode: str, dry_run: bool = False) -> None:
        """Log experiment run to history file.

        Args:
            experiment_file: Path to experiment config file
            experiment: Experiment configuration
            job_ids: List of submitted job IDs
            runs: List of (model, dataset, seed) tuples that were executed
            mode: Execution mode (e.g., "sbatch [direct]", "local", etc.)
            dry_run: Whether this was a dry run
        """
        if dry_run:
            return  # Don't log dry runs

        # Convert to absolute path and then make relative
        abs_exp_file = experiment_file.resolve()
        try:
            rel_path = abs_exp_file.relative_to(self.project_root)
        except ValueError:
            # If path is not relative to project root, use the full path
            rel_path = abs_exp_file

        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_file": str(rel_path),
            "mode": mode,
            "num_jobs": len(job_ids),
            "job_ids": job_ids,
            "config": {
                "models": experiment.models,
                "datasets": experiment.datasets,
                "seeds": experiment.seeds,
                "pos_weight": experiment.pos_weight,
                "loss_type": getattr(experiment, 'loss_type', 'bce'),
                "epoch": experiment.epoch,
                "out_suffix": experiment.out_suffix,
                "mode": experiment.mode,
                "threshold_method": experiment.threshold_method,
                "threshold_metric": experiment.threshold_metric,
                "learning_rate": experiment.learning_rate,
                "dropout_probability": experiment.dropout_probability,
                "wandb_project": experiment.wandb_project if experiment.use_wandb else None,
            },
            "runs": [
                {"model": model, "dataset": dataset, "seed": seed}
                for model, dataset, seed in runs
            ]
        }

        # Append to history file (JSONL format)
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(history_entry) + '\n')

    def run_experiment(self, experiment: ExperimentConfig,
                      use_sbatch: bool = True,
                      legacy_mode: bool = False,
                      dry_run: bool = False,
                      fix_missing: bool = False,
                      anonymized: bool = False,
                      experiment_file: Optional[Path] = None) -> None:
        """Run an experiment.

        Args:
            experiment: Experiment configuration
            use_sbatch: Whether to use sbatch for job submission
            legacy_mode: If True (with use_sbatch), use old bash scripts with env vars.
                        If False (with use_sbatch), use sbatch --wrap with direct Python calls.
            dry_run: If True, print commands without executing
            fix_missing: If True, only run experiments that are missing from results directory
            anonymized: If True, run anonymized versions of the datasets. e.g. juliet_full_dataset_anonymized.jsonl
        """
        # Validate
        errors = self.config_loader.validate_experiment(experiment)
        if errors:
            print("Experiment validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

        # Load configs
        models = self.config_loader.load_models()
        datasets = self.config_loader.load_datasets()

        # Determine what to run
        if fix_missing:

            # Load models config for directory mapping
            from find_missing_experiments import load_config_files
            models_config, _ = load_config_files(self.config_dir)

            # Convert ExperimentConfig to dict format for compatibility
            exp_dict = {
                "models": experiment.models,
                "datasets": experiment.datasets,
                "seeds": experiment.seeds,
                "out_suffix": experiment.out_suffix
            }

            # Get expected and actual experiments
            expected = get_expected_experiments(exp_dict)
            print(f"Expected {len(expected)} experiment combinations")

            print(f"Scanning results directory: {self.models_dir}")
            actual = get_actual_experiments(self.models_dir, models_config, experiment.out_suffix)
            print(f"Found {len(actual)} completed experiments")

            # Find missing
            missing = find_missing_experiments(expected, actual)

            if not missing:
                print("\n✓ All experiments already completed!")
                return

            print(f"\nFound {len(missing)} missing experiments")
            runs_to_execute = missing
        else:
            # Generate Cartesian product
            runs_to_execute = [
                (model_name, dataset_name, seed)
                for model_name in experiment.models
                for dataset_name in experiment.datasets
                for seed in experiment.seeds
            ]
            print(f"Running experiment with {len(experiment.models)} model(s), "
                  f"{len(experiment.datasets)} dataset(s), {len(experiment.seeds)} seed(s)")

        mode_desc = "legacy (bash)" if legacy_mode else "direct (Python)"
        if use_sbatch:
            print(f"Execution mode: sbatch [{mode_desc}]")
        else:
            print(f"Execution mode: local")

        if dry_run:
            print("\n=== DRY RUN MODE ===\n")

        # Execute jobs
        job_count = 0
        job_ids = []
        for model_name, dataset_name, seed in runs_to_execute:
            model = models[model_name]
            dataset = datasets[dataset_name]
            job_count += 1

            if use_sbatch:

                if legacy_mode:
                    # Legacy mode: use bash scripts with env vars
                    env = self._setup_env(model, dataset, experiment, seed)
                    script_name = f"train_split.sh" if experiment.mode == "train" else "test_split.sh"
                    script_path = self.scripts_dir / script_name

                    sbatch_cmd = self._build_sbatch_command(dataset, model_name, experiment, seed, legacy_mode=True, model=model, anonymized=anonymized)
                    sbatch_cmd.append(str(script_path))

                    if dry_run:
                        print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                        print(f"  Command: {' '.join(sbatch_cmd)}")
                        print(f"  Script: {script_path}")
                        print(f"  Env vars: model_name={model.model_name}, dataset_name={dataset_name}, seed={seed}")
                        print()
                    else:
                        result = subprocess.run(sbatch_cmd, env=env, check=True, capture_output=True, text=True)
                        # Extract job ID from "Submitted batch job 123456"
                        job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
                        job_ids.append(job_id)
                        print(result.stdout.strip())
                else:
                    # Direct mode: use sbatch --wrap with Python command
                    sbatch_cmd = self._build_sbatch_command(dataset, model_name, experiment, seed, legacy_mode=False, model=model, anonymized=anonymized)
                    wrap_cmd = self._build_sbatch_wrap_command(model, dataset, experiment, seed, model_name, anonymized)

                    sbatch_cmd.append("--wrap")
                    sbatch_cmd.append(wrap_cmd)

                    if dry_run:
                        print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                        print(f"  sbatch: {' '.join(sbatch_cmd[:-2])}")  # Print sbatch args
                        print(f"  --wrap: {wrap_cmd}")
                        print()
                    else:
                        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
                        # Extract job ID from "Submitted batch job 123456"
                        job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
                        job_ids.append(job_id)
                        print(result.stdout.strip())
            else:
                # Run directly with Python (no sbatch)
                env = self._setup_env(model, dataset, experiment, seed)
                python_cmd = self._build_python_command(model, dataset, experiment, seed, model_name, anonymized)

                if dry_run:
                    print(f"Job {job_count}: {model_name} × {dataset_name} × seed={seed}")
                    print(f"  Command: {' '.join(python_cmd)}")
                    print()
                else:
                    # Activate conda environment and run
                    conda_activate = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ensemble"
                    full_cmd = f"{conda_activate} && {' '.join(python_cmd)}"
                    subprocess.run(full_cmd, shell=True, env=env, check=True)

        if not dry_run:
            print(f"Submitted {job_count} job(s)")

            # Log run history
            if experiment_file:
                mode_str = f"sbatch [{mode_desc}]" if use_sbatch else "local"
                self._log_run_history(experiment_file, experiment, job_ids, runs_to_execute, mode_str, dry_run)
        else:
            print(f"=== Would submit {job_count} job(s) ===")


def find_project_root() -> Path:
    """Find the project root using git."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True
    )
    return Path(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Run vulnerability detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution modes:
  Default:      sbatch with direct Python calls (sbatch --wrap)
  --legacy:     sbatch with legacy bash scripts (train_split.sh/test_split.sh)
  --no-sbatch:  Run locally without sbatch

Examples:
  python runner.py train_linevul_all                    # Direct Python via sbatch
  python runner.py train_linevul_all --legacy           # Legacy bash scripts via sbatch
  python runner.py train_linevul_all --dry-run          # Preview commands
  python runner.py train_linevul_all --fix-missing      # Only run missing experiments
  python runner.py train_linevul_all --no-sbatch        # Run locally
        """
    )
    parser.add_argument("experiment", nargs="?", help="Path to experiment config JSON file")
    parser.add_argument("--no-sbatch", action="store_true",
                       help="Run directly without sbatch")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy bash scripts with sbatch (instead of direct Python)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--fix-missing", action="store_true",
                       help="Only run experiments that are missing from results directory")
    parser.add_argument("--anonymized", action="store_true",
                       help="Run anonymized versions of the datasets. e.g. juliet_full_dataset_anonymized.jsonl")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")
    parser.add_argument("--list-experiments", action="store_true",
                       help="List available datasets")

    args = parser.parse_args()

    project_root = find_project_root()
    runner = ExperimentRunner(project_root)

    # List commands
    if args.list_models:
        models = runner.config_loader.load_models()
        print("Available models:")
        for name, config in models.items():
            print(f"  {name}: {config.model_name} ({config.model_type})")
        return

    if args.list_datasets:
        datasets = runner.config_loader.load_datasets()
        print("Available datasets:")
        for name, config in datasets.items():
            print(f"  {name}: {config.size}, {config.time_hours}h on {config.gpu}")
        print(f"\nDataset groups: small, big, all")
        return
    if args.list_experiments:
        p = Path(project_root / "scripts" / "config" / "experiments" )
        files = [item.name for item in p.iterdir() if item.is_file()]
        print("Available experiments:")
        for file in files:
            print("\t", file)
        return


    if not args.experiment:
        parser.print_help()
        sys.exit(1)

    # Load and run experiment
    experiment_file = Path(args.experiment)
    if not experiment_file.exists():
        # Try relative to config/experiments
        experiment_file = project_root / "scripts" / "config" / "experiments" / args.experiment
        if not experiment_file.suffix == ".json":
            experiment_file = experiment_file.with_suffix(".json")

    if not experiment_file.exists():
        print(f"Error: Experiment file not found: {args.experiment}", file=sys.stderr)
        sys.exit(1)

    experiment = runner.config_loader.load_experiment(experiment_file)
    runner.run_experiment(
        experiment,
        use_sbatch=not args.no_sbatch,
        legacy_mode=args.legacy,
        dry_run=args.dry_run,
        fix_missing=args.fix_missing,
        anonymized=args.anonymized,
        experiment_file=experiment_file
    )


if __name__ == "__main__":
    main()
