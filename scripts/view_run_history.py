#!/usr/bin/env python3
"""View and query the experiment run history."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to human-readable format."""
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def view_history(history_file: Path, last_n: Optional[int] = None,
                 experiment_filter: Optional[str] = None,
                 verbose: bool = False) -> None:
    """View run history.

    Args:
        history_file: Path to run_history.jsonl
        last_n: Show only last N runs
        experiment_filter: Filter by experiment file name (partial match)
        verbose: Show detailed information
    """
    if not history_file.exists():
        print(f"No run history found at {history_file}")
        return

    # Read all history entries
    entries = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Filter by experiment name if requested
    if experiment_filter:
        entries = [e for e in entries if experiment_filter in e['experiment_file']]

    # Get last N entries if requested
    if last_n:
        entries = entries[-last_n:]

    if not entries:
        print("No matching runs found")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(entries)} run(s)")
    print(f"{'='*80}\n")

    for i, entry in enumerate(entries, 1):
        timestamp = format_timestamp(entry['timestamp'])
        exp_file = entry['experiment_file']
        mode = entry['mode']
        num_jobs = entry['num_jobs']

        print(f"[{i}] {timestamp} - {exp_file}")
        print(f"    Mode: {mode}, Jobs: {num_jobs}")

        if verbose:
            config = entry['config']
            print(f"    Models: {', '.join(config['models'])}")
            print(f"    Datasets: {', '.join(config['datasets'])}")
            print(f"    Seeds: {config['seeds']}")
            print(f"    Loss: {config.get('loss_type', 'bce')}, Pos Weight: {config['pos_weight']}")
            print(f"    Epochs: {config['epoch']}, LR: {config['learning_rate']}")
            print(f"    Threshold Metric: {config['threshold_metric']}")
            if config.get('wandb_project'):
                print(f"    Wandb Project: {config['wandb_project']}")

            if entry.get('job_ids'):
                job_ids_str = ', '.join(entry['job_ids'][:5])
                if len(entry['job_ids']) > 5:
                    job_ids_str += f"... ({len(entry['job_ids']) - 5} more)"
                print(f"    Job IDs: {job_ids_str}")

        print()


def show_stats(history_file: Path) -> None:
    """Show statistics about run history."""
    if not history_file.exists():
        print(f"No run history found at {history_file}")
        return

    entries = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("No runs in history")
        return

    # Gather statistics
    total_runs = len(entries)
    total_jobs = sum(e['num_jobs'] for e in entries)
    experiments = set(e['experiment_file'] for e in entries)
    models = set()
    datasets = set()
    loss_types = {}

    for e in entries:
        config = e['config']
        models.update(config['models'])
        datasets.update(config['datasets'])
        loss_type = config.get('loss_type', 'bce')
        loss_types[loss_type] = loss_types.get(loss_type, 0) + 1

    # Get date range
    first_run = datetime.fromisoformat(entries[0]['timestamp'])
    last_run = datetime.fromisoformat(entries[-1]['timestamp'])

    print("\n" + "="*60)
    print("Run History Statistics")
    print("="*60)
    print(f"Total experiment runs: {total_runs}")
    print(f"Total jobs submitted: {total_jobs}")
    print(f"Unique experiments: {len(experiments)}")
    print(f"Date range: {first_run.strftime('%Y-%m-%d')} to {last_run.strftime('%Y-%m-%d')}")
    print(f"\nModels used: {', '.join(sorted(models))}")
    print(f"Datasets used: {', '.join(sorted(datasets))}")
    print(f"\nLoss functions:")
    for loss_type, count in sorted(loss_types.items()):
        print(f"  {loss_type}: {count} runs")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View experiment run history",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-n", "--last", type=int, metavar="N",
                       help="Show only last N runs")
    parser.add_argument("-f", "--filter", type=str, metavar="NAME",
                       help="Filter by experiment file name (partial match)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics instead of listing runs")
    parser.add_argument("--history-file", type=Path,
                       help="Path to run_history.jsonl (default: scripts/run_history.jsonl)")

    args = parser.parse_args()

    # Find history file
    if args.history_file:
        history_file = args.history_file
    else:
        # Find project root
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True
        )
        project_root = Path(result.stdout.strip())
        history_file = project_root / "scripts" / "run_history.jsonl"

    if args.stats:
        show_stats(history_file)
    else:
        view_history(history_file, args.last, args.filter, args.verbose)


if __name__ == "__main__":
    main()
