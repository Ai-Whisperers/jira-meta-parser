"""Utility functions for training pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def load_artifacts(artifacts_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all required artifacts for training.

    Args:
        artifacts_dir: Path to artifacts directory

    Returns:
        Dictionary with features, embeddings, weak_labels, validation dataframes
    """
    artifacts_path = Path(artifacts_dir)

    # Find latest artifacts
    features_path = _get_latest_artifact(artifacts_path / "features", "variability_features", "parquet")
    embeddings_path = _get_latest_artifact(artifacts_path / "embeddings", "embeddings", "parquet")
    weak_labels_path = _get_latest_artifact(artifacts_path / "labels", "weak_labels", "csv")
    validation_path = _get_latest_artifact(artifacts_path / "validation", "backbone_report", "csv")

    if not all([features_path, embeddings_path, weak_labels_path, validation_path]):
        raise FileNotFoundError(
            "Could not find all required artifacts. "
            "Please run the pipeline first to generate artifacts."
        )

    # Load artifacts
    features_df = pd.read_parquet(features_path)
    embeddings_df = pd.read_parquet(embeddings_path)
    weak_labels_df = pd.read_csv(weak_labels_path)
    validation_df = pd.read_csv(validation_path)

    return {
        "features": features_df,
        "embeddings": embeddings_df,
        "weak_labels": weak_labels_df,
        "validation": validation_df,
    }


def _get_latest_artifact(directory: Path, name: str, format: str) -> Optional[Path]:
    """Get latest version of an artifact.

    Args:
        directory: Directory to search
        name: Artifact base name
        format: File format

    Returns:
        Path to latest artifact or None
    """
    if not directory.exists():
        return None

    # Look for versioned files first
    pattern = f"{name}_*.{format}"
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)

    if files:
        return files[-1]

    # Fall back to non-versioned
    unversioned = directory / f"{name}.{format}"
    if unversioned.exists():
        return unversioned

    return None


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: List[int],
    k_values: List[int] = [10, 20, 50],
) -> Dict[str, float]:
    """Compute ranking evaluation metrics.

    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        groups: Group sizes for ranking
        k_values: K values for NDCG@K

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Compute NDCG@K for each group
    start_idx = 0
    ndcg_scores = {k: [] for k in k_values}

    for group_size in groups:
        end_idx = start_idx + group_size

        y_true_group = y_true[start_idx:end_idx]
        y_pred_group = y_pred[start_idx:end_idx]

        # Compute NDCG@K
        for k in k_values:
            if len(y_true_group) >= k:
                score = ndcg_score(
                    y_true_group.reshape(1, -1),
                    y_pred_group.reshape(1, -1),
                    k=k,
                )
                ndcg_scores[k].append(score)

        start_idx = end_idx

    # Average NDCG@K across groups
    for k in k_values:
        if ndcg_scores[k]:
            metrics[f"ndcg@{k}"] = np.mean(ndcg_scores[k])
        else:
            metrics[f"ndcg@{k}"] = 0.0

    # Overall NDCG (no K limit)
    ndcg_all = []
    start_idx = 0
    for group_size in groups:
        end_idx = start_idx + group_size
        y_true_group = y_true[start_idx:end_idx]
        y_pred_group = y_pred[start_idx:end_idx]

        score = ndcg_score(
            y_true_group.reshape(1, -1),
            y_pred_group.reshape(1, -1),
        )
        ndcg_all.append(score)
        start_idx = end_idx

    metrics["ndcg"] = np.mean(ndcg_all)

    # Spearman correlation (measure rank correlation)
    from scipy.stats import spearmanr
    if len(y_true) > 1:
        correlation, _ = spearmanr(y_true, y_pred)
        metrics["spearman"] = correlation
    else:
        metrics["spearman"] = 0.0

    # Kendall's Tau (another rank correlation measure)
    from scipy.stats import kendalltau
    if len(y_true) > 1:
        tau, _ = kendalltau(y_true, y_pred)
        metrics["kendall_tau"] = tau
    else:
        metrics["kendall_tau"] = 0.0

    return metrics


def save_training_report(
    report: Dict,
    output_path: str,
):
    """Save training report to file.

    Args:
        report: Report dictionary
        output_path: Output file path
    """
    import json

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def plot_training_curves(
    train_history: Dict,
    output_path: str,
):
    """Plot training curves (if matplotlib available).

    Args:
        train_history: Training history dictionary
        output_path: Output file path
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot metrics
        for metric_name, values in train_history.items():
            ax.plot(values, label=metric_name)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Metric Value")
        ax.set_title("Training Curves")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except ImportError:
        # Matplotlib not available
        pass


def compare_with_heuristic(
    y_true: np.ndarray,
    y_pred_ml: np.ndarray,
    y_pred_heuristic: np.ndarray,
    groups: List[int],
) -> Dict[str, Dict[str, float]]:
    """Compare ML model with heuristic baseline.

    Args:
        y_true: True relevance scores
        y_pred_ml: ML model predictions
        y_pred_heuristic: Heuristic baseline predictions
        groups: Group sizes

    Returns:
        Dictionary with comparison metrics
    """
    ml_metrics = compute_ranking_metrics(y_true, y_pred_ml, groups)
    heuristic_metrics = compute_ranking_metrics(y_true, y_pred_heuristic, groups)

    comparison = {
        "ml": ml_metrics,
        "heuristic": heuristic_metrics,
        "improvement": {},
    }

    # Compute improvement percentages
    for metric_name in ml_metrics:
        if metric_name in heuristic_metrics:
            baseline = heuristic_metrics[metric_name]
            if baseline != 0:
                improvement = ((ml_metrics[metric_name] - baseline) / baseline) * 100
                comparison["improvement"][metric_name] = improvement

    return comparison
