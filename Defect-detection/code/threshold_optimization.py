"""
Threshold optimization methods for binary classification.

Implements both simple grid search and GHOST (Generalized tHreshOld ShifTing)
methods for finding optimal decision thresholds.

References:
- GHOST: Reymond, J.-L., et al. (2021). "GHOST: Adjusting the decision threshold
  to maximize the Matthews correlation coefficient." J. Chem. Inf. Model. 61(4): 1534-1540.
  https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160
  https://github.com/rinikerlab/GHOST
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics_with_threshold(threshold: float, logits: np.ndarray,
                                     labels: np.ndarray) -> Dict:
    """
    Calculate binary classification metrics for a given threshold.

    Args:
        threshold: Decision threshold value
        logits: Predicted probabilities (N, 1) or (N,) array
        labels: True binary labels

    Returns:
        Dictionary containing all classification metrics
    """
    # Handle both (N, 1) and (N,) shaped arrays
    if len(logits.shape) > 1:
        preds = logits[:, 0] > threshold
    else:
        preds = logits > threshold

    true_vul = ((preds == 1) & (labels == 1)).sum()
    false_vul = ((preds == 1) & (labels == 0)).sum()
    false_non = ((preds == 0) & (labels == 1)).sum()
    true_non = ((preds == 0) & (labels == 0)).sum()

    precision = true_vul / (true_vul + false_vul) if (true_vul + false_vul) > 0 else 0
    recall = true_vul / (true_vul + false_non) if (true_vul + false_non) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_vul + true_non) / len(labels)

    # Calculate specificity (true negative rate)
    specificity = true_non / (true_non + false_vul) if (true_non + false_vul) > 0 else 0

    # Calculate Matthews Correlation Coefficient (MCC)
    mcc_numerator = (true_vul * true_non) - (false_vul * false_non)
    mcc_denominator = np.sqrt((true_vul + false_vul) * (true_vul + false_non) *
                              (true_non + false_vul) * (true_non + false_non))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    # Cohen's Kappa
    total = len(labels)
    p_o = (true_vul + true_non) / total  # Observed agreement
    p_e = ((true_vul + false_vul) * (true_vul + false_non) +
           (true_non + false_non) * (true_non + false_vul)) / (total * total)  # Expected agreement
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "mcc": mcc,
        "kappa": kappa,
        "true_pos": int(true_vul),
        "false_pos": int(false_vul),
        "false_neg": int(false_non),
        "true_neg": int(true_non)
    }


def grid_search_threshold(logits: np.ndarray, labels: np.ndarray,
                         thresholds: np.ndarray = None,
                         optimization_metric: str = "f1",
                         min_recall: float = 0.0,
                         precision_weight: float = 1.0,
                         min_precision: float = 0.0,
                         allow_accuracy_decrease: bool = True) -> Tuple[float, Dict, List[Dict]]:
    """
    Find optimal threshold using simple grid search.

    This method evaluates each threshold once on the full dataset and selects
    the threshold that maximizes the chosen optimization metric.

    Args:
        logits: Predicted probabilities
        labels: True binary labels
        thresholds: Array of threshold values to test (default: 0.1 to 0.9, step 0.02)
        optimization_metric: Metric to optimize ('f1', 'precision', 'mcc', 'kappa')
        min_recall: Minimum recall constraint for precision optimization
        precision_weight: Weight for precision in multi-objective optimization

    Returns:
        Tuple of (optimal_threshold, best_metrics, all_threshold_results)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    best_score = -1
    best_threshold = 0.5
    best_metrics = None
    threshold_results = []

    logger.info(f"Grid search: Testing {len(thresholds)} thresholds from {thresholds.min():.2f} to {thresholds.max():.2f}")

    for threshold in thresholds:
        metrics = calculate_metrics_with_threshold(threshold, logits, labels)
        threshold_results.append(metrics)

        # Calculate score based on optimization metric
        if optimization_metric == "f1":
            score = metrics["f1"]
        elif optimization_metric == "precision":
            if metrics["recall"] >= min_recall:
                score = (precision_weight * metrics["precision"] +
                        metrics["recall"] + metrics["f1"])
            else:
                score = -1
        elif optimization_metric == "mcc":
            score = metrics["mcc"]
        elif optimization_metric == "kappa":
            score = metrics["kappa"]
        else:
            score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        logger.warning("No optimal threshold found. Using default threshold 0.5")
        best_metrics = calculate_metrics_with_threshold(0.5, logits, labels)
        best_threshold = 0.5

    logger.info(f"Grid search optimal: threshold={best_threshold:.3f}, F1={best_metrics['f1']:.4f}")

    return best_threshold, best_metrics, threshold_results


def ghost_threshold_optimization(logits: np.ndarray, labels: np.ndarray,
                                 thresholds: np.ndarray = None,
                                 optimization_metric: str = "kappa",
                                 n_subsets: int = 100,
                                 subset_size: float = 0.8,
                                 with_replacement: bool = False,
                                 random_seed: Optional[int] = None) -> Tuple[float, Dict, Dict]:
    """
    Find optimal threshold using GHOST (Generalized tHreshOld ShifTing) method.

    GHOST uses bootstrap aggregation to find a robust optimal threshold. Instead of
    evaluating on the full dataset once, it:
    1. Creates N random subsets of the training data
    2. Evaluates each threshold on all subsets
    3. Selects the threshold with best median performance across subsets

    This provides more stable threshold selection by accounting for variance in
    the data distribution.

    Reference:
        Reymond, J.-L., et al. (2021). "GHOST: Adjusting the decision threshold
        to maximize the Matthews correlation coefficient."
        J. Chem. Inf. Model. 61(4): 1534-1540.

    Args:
        logits: Predicted probabilities
        labels: True binary labels
        thresholds: Array of threshold values to test (default: 0.1 to 0.9, step 0.02)
        optimization_metric: Metric to optimize ('kappa', 'mcc', 'f1', 'roc')
        n_subsets: Number of bootstrap subsets to create
        subset_size: Proportion (0-1) or count (>1) of samples per subset
        with_replacement: Whether to sample with replacement (bootstrap)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (optimal_threshold, best_metrics, ghost_statistics)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = len(labels)
    if subset_size <= 1.0:
        subset_n = int(n_samples * subset_size)
    else:
        subset_n = int(subset_size)

    logger.info(f"GHOST: Testing {len(thresholds)} thresholds on {n_subsets} subsets "
               f"(size={subset_n}/{n_samples}, metric={optimization_metric})")

    # Store results for each threshold across all subsets
    threshold_scores = {threshold: [] for threshold in thresholds}

    # Generate subsets and evaluate
    for subset_idx in range(n_subsets):
        # Sample subset
        if with_replacement:
            indices = np.random.choice(n_samples, size=subset_n, replace=True)
        else:
            indices = np.random.choice(n_samples, size=subset_n, replace=False)

        subset_logits = logits[indices]
        subset_labels = labels[indices]

        # Evaluate all thresholds on this subset
        for threshold in thresholds:
            metrics = calculate_metrics_with_threshold(threshold, subset_logits, subset_labels)

            # Calculate score based on optimization metric
            if optimization_metric == "kappa":
                score = metrics["kappa"]
            elif optimization_metric == "mcc":
                score = metrics["mcc"]
            elif optimization_metric == "f1":
                score = metrics["f1"]
            elif optimization_metric == "roc":
                # ROC-based optimization: geometric mean of sensitivity and specificity
                sensitivity = metrics["recall"]
                specificity = metrics["specificity"]
                score = (2 * sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0
            else:
                score = metrics["kappa"]

            threshold_scores[threshold].append(score)

    # Calculate median and std for each threshold
    threshold_statistics = {}
    for threshold in thresholds:
        scores = threshold_scores[threshold]
        threshold_statistics[threshold] = {
            "median": np.median(scores),
            "std": np.std(scores),
            "mean": np.mean(scores),
            "min": np.min(scores),
            "max": np.max(scores)
        }

    # Select threshold with best median performance
    best_threshold = max(threshold_statistics.keys(),
                        key=lambda t: threshold_statistics[t]["median"])

    # Calculate final metrics on full dataset with optimal threshold
    best_metrics = calculate_metrics_with_threshold(best_threshold, logits, labels)

    # Prepare statistics report
    ghost_stats = {
        "optimal_threshold": best_threshold,
        "n_subsets": n_subsets,
        "subset_size": subset_n,
        "optimization_metric": optimization_metric,
        "optimal_median_score": threshold_statistics[best_threshold]["median"],
        "optimal_std_score": threshold_statistics[best_threshold]["std"],
        "threshold_statistics": threshold_statistics
    }

    logger.info(f"GHOST optimal: threshold={best_threshold:.3f}, "
               f"median_{optimization_metric}={threshold_statistics[best_threshold]['median']:.4f}±"
               f"{threshold_statistics[best_threshold]['std']:.4f}, "
               f"F1={best_metrics['f1']:.4f}")

    return best_threshold, best_metrics, ghost_stats


def compare_threshold_methods(logits: np.ndarray, labels: np.ndarray,
                              thresholds: np.ndarray = None,
                              optimization_metric: str = "f1",
                              ghost_metric: str = "kappa",
                              n_subsets: int = 100,
                              random_seed: Optional[int] = None) -> Dict:
    """
    Compare grid search and GHOST threshold optimization methods.

    Args:
        logits: Predicted probabilities
        labels: True binary labels
        thresholds: Array of threshold values to test
        optimization_metric: Metric for grid search optimization
        ghost_metric: Metric for GHOST optimization
        n_subsets: Number of subsets for GHOST
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing results from both methods and comparison statistics
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    logger.info("=" * 80)
    logger.info("Comparing threshold optimization methods")
    logger.info("=" * 80)

    # Grid search
    logger.info("\n1. Grid Search Method")
    grid_threshold, grid_metrics, grid_results = grid_search_threshold(
        logits, labels, thresholds, optimization_metric
    )

    # GHOST
    logger.info("\n2. GHOST Method")
    ghost_threshold, ghost_metrics, ghost_stats = ghost_threshold_optimization(
        logits, labels, thresholds, ghost_metric, n_subsets, random_seed=random_seed
    )

    # Default 0.5 threshold for comparison
    default_metrics = calculate_metrics_with_threshold(0.5, logits, labels)

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("Comparison Summary")
    logger.info("=" * 80)
    logger.info(f"Default (0.5):    F1={default_metrics['f1']:.4f}, "
               f"Precision={default_metrics['precision']:.4f}, "
               f"Recall={default_metrics['recall']:.4f}")
    logger.info(f"Grid Search:      threshold={grid_threshold:.3f}, "
               f"F1={grid_metrics['f1']:.4f}, "
               f"Precision={grid_metrics['precision']:.4f}, "
               f"Recall={grid_metrics['recall']:.4f}")
    logger.info(f"GHOST:            threshold={ghost_threshold:.3f}, "
               f"F1={ghost_metrics['f1']:.4f}, "
               f"Precision={ghost_metrics['precision']:.4f}, "
               f"Recall={ghost_metrics['recall']:.4f}")

    return {
        "default_threshold": 0.5,
        "default_metrics": default_metrics,
        "grid_search": {
            "threshold": grid_threshold,
            "metrics": grid_metrics,
            "all_results": grid_results
        },
        "ghost": {
            "threshold": ghost_threshold,
            "metrics": ghost_metrics,
            "statistics": ghost_stats
        },
        "thresholds_tested": thresholds.tolist()
    }
