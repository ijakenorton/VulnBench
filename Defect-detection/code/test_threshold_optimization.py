"""
Test script for threshold optimization methods.
Verifies that grid search and GHOST implementations work correctly.
"""

import numpy as np
import sys
from threshold_optimization import (
    grid_search_threshold,
    ghost_threshold_optimization,
    compare_threshold_methods,
    calculate_metrics_with_threshold
)


def generate_synthetic_data(n_samples=1000, pos_ratio=0.1, seed=42):
    """Generate synthetic binary classification data for testing."""
    np.random.seed(seed)

    # Generate imbalanced labels
    n_pos = int(n_samples * pos_ratio)
    n_neg = n_samples - n_pos
    labels = np.array([1] * n_pos + [0] * n_neg)

    # Generate logits with some separation
    # Positive samples: higher mean
    # Negative samples: lower mean
    pos_logits = np.random.normal(0.6, 0.2, n_pos)
    neg_logits = np.random.normal(0.3, 0.2, n_neg)

    # Clip to [0, 1] range
    pos_logits = np.clip(pos_logits, 0, 1)
    neg_logits = np.clip(neg_logits, 0, 1)

    # Combine and shuffle
    logits = np.concatenate([pos_logits, neg_logits])
    indices = np.random.permutation(n_samples)

    logits = logits[indices].reshape(-1, 1)  # Shape: (n_samples, 1)
    labels = labels[indices]

    return logits, labels


def test_calculate_metrics():
    """Test basic metrics calculation."""
    print("="*80)
    print("Test 1: Basic Metrics Calculation")
    print("="*80)

    # Simple test case
    logits = np.array([[0.1], [0.3], [0.7], [0.9]])
    labels = np.array([0, 0, 1, 1])

    metrics = calculate_metrics_with_threshold(0.5, logits, labels)

    print(f"Threshold: {metrics['threshold']}")
    print(f"TP={metrics['true_pos']}, FP={metrics['false_pos']}, "
          f"FN={metrics['false_neg']}, TN={metrics['true_neg']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")

    # Expected: TP=2, FP=0, FN=0, TN=2 (perfect classification at 0.5)
    assert metrics['true_pos'] == 2, "Expected 2 true positives"
    assert metrics['false_pos'] == 0, "Expected 0 false positives"
    assert metrics['precision'] == 1.0, "Expected perfect precision"
    assert metrics['recall'] == 1.0, "Expected perfect recall"
    assert metrics['f1'] == 1.0, "Expected perfect F1"

    print("✓ Basic metrics calculation passed\n")


def test_grid_search():
    """Test grid search threshold optimization."""
    print("="*80)
    print("Test 2: Grid Search Threshold Optimization")
    print("="*80)

    logits, labels = generate_synthetic_data(n_samples=500, pos_ratio=0.15)

    print(f"Dataset: {len(labels)} samples, {(labels==1).sum()} positive ({100*(labels==1).sum()/len(labels):.1f}%)")

    # Test with different metrics
    for metric in ["f1", "kappa", "mcc"]:
        print(f"\nOptimizing for {metric}...")
        threshold, metrics, results = grid_search_threshold(
            logits, labels,
            optimization_metric=metric
        )

        print(f"  Optimal threshold: {threshold:.3f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Kappa: {metrics['kappa']:.4f}")

        assert len(results) > 0, "Should have evaluated multiple thresholds"
        assert 0 <= threshold <= 1, "Threshold should be in [0, 1]"
        assert 0 <= metrics['f1'] <= 1, "F1 should be in [0, 1]"

    print("\n✓ Grid search optimization passed\n")


def test_ghost():
    """Test GHOST threshold optimization."""
    print("="*80)
    print("Test 3: GHOST Threshold Optimization")
    print("="*80)

    logits, labels = generate_synthetic_data(n_samples=500, pos_ratio=0.15, seed=123)

    print(f"Dataset: {len(labels)} samples, {(labels==1).sum()} positive ({100*(labels==1).sum()/len(labels):.1f}%)")

    # Test with different metrics and smaller n_subsets for speed
    for metric in ["kappa", "mcc", "f1"]:
        print(f"\nOptimizing for {metric} with GHOST...")
        threshold, metrics, stats = ghost_threshold_optimization(
            logits, labels,
            optimization_metric=metric,
            n_subsets=20,  # Reduced for testing speed
            subset_size=0.8,
            random_seed=42
        )

        print(f"  Optimal threshold: {threshold:.3f}")
        print(f"  Median {metric}: {stats['optimal_median_score']:.4f} ± {stats['optimal_std_score']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  Kappa: {metrics['kappa']:.4f}")

        assert 0 <= threshold <= 1, "Threshold should be in [0, 1]"
        assert 0 <= metrics['f1'] <= 1, "F1 should be in [0, 1]"
        assert stats['n_subsets'] == 20, "Should have used 20 subsets"

    print("\n✓ GHOST optimization passed\n")


def test_comparison():
    """Test comparison of both methods."""
    print("="*80)
    print("Test 4: Method Comparison")
    print("="*80)

    logits, labels = generate_synthetic_data(n_samples=500, pos_ratio=0.15, seed=456)

    print(f"Dataset: {len(labels)} samples, {(labels==1).sum()} positive ({100*(labels==1).sum()/len(labels):.1f}%)")

    comparison = compare_threshold_methods(
        logits, labels,
        optimization_metric="f1",
        ghost_metric="kappa",
        n_subsets=20,  # Reduced for testing speed
        random_seed=42
    )

    print("\nComparison Results:")
    print(f"Default (0.5) - F1: {comparison['default_metrics']['f1']:.4f}")
    print(f"Grid Search - Threshold: {comparison['grid_search']['threshold']:.3f}, "
          f"F1: {comparison['grid_search']['metrics']['f1']:.4f}")
    print(f"GHOST - Threshold: {comparison['ghost']['threshold']:.3f}, "
          f"F1: {comparison['ghost']['metrics']['f1']:.4f}")

    assert 'default_metrics' in comparison
    assert 'grid_search' in comparison
    assert 'ghost' in comparison

    print("\n✓ Method comparison passed\n")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("="*80)
    print("Test 5: Edge Cases")
    print("="*80)

    # All positive labels - should select low threshold for high recall
    print("Testing all positive labels...")
    logits = np.random.rand(100, 1)
    labels = np.ones(100)
    threshold, metrics, _ = grid_search_threshold(logits, labels)
    print(f"  Threshold: {threshold:.3f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    # With all positive labels, any threshold that classifies anything as positive will work
    # The optimization will balance between catching all positives and avoiding FP (but there are no negatives)
    assert metrics['recall'] > 0, "Should have some recall"
    assert 0 <= metrics['f1'] <= 1, "F1 should be valid"

    # All negative labels
    print("Testing all negative labels...")
    labels = np.zeros(100)
    threshold, metrics, _ = grid_search_threshold(logits, labels)
    print(f"  Threshold: {threshold:.3f}, Specificity: {metrics['specificity']:.4f}")
    # With all negative labels, high threshold gives high specificity
    assert metrics['specificity'] > 0, "Should have some specificity"
    assert 0 <= metrics['f1'] <= 1, "F1 should be valid (may be 0 if no TP)"

    # Very imbalanced (1% positive)
    print("Testing highly imbalanced data (1% positive)...")
    logits, labels = generate_synthetic_data(n_samples=1000, pos_ratio=0.01, seed=789)
    threshold, metrics, _ = grid_search_threshold(logits, labels, optimization_metric="mcc")
    print(f"  Threshold: {threshold:.3f}, MCC: {metrics['mcc']:.4f}, F1: {metrics['f1']:.4f}")
    assert -1 <= metrics['mcc'] <= 1, "MCC should be in [-1, 1]"
    assert 0 <= metrics['f1'] <= 1, "F1 should be valid"

    print("\n✓ Edge cases passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION TEST SUITE")
    print("="*80 + "\n")

    try:
        test_calculate_metrics()
        test_grid_search()
        test_ghost()
        test_comparison()
        test_edge_cases()

        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TEST FAILED ✗")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
