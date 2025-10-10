"""Training script for LightGBM LambdaMART ranker using weak labels."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

# Import from main codebase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ranker import LambdaMARTRanker, create_ranker
from src.utils.config import Config
from training.config import load_training_config
from training.utils import (
    load_artifacts,
    save_training_report,
    compute_ranking_metrics,
    plot_training_curves,
)


class RankerTrainer:
    """Handles training of LightGBM LambdaMART ranker."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer.

        Args:
            config_path: Optional path to config file
        """
        self.training_config = load_training_config(config_path)
        self.base_config = Config(config_path).to_dict()
        self.ranker = create_ranker(self.base_config)

        # Training state
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

    def load_data(
        self,
        artifacts_dir: str = "./artifacts",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed artifacts.

        Args:
            artifacts_dir: Path to artifacts directory

        Returns:
            Tuple of (features_df, embeddings_df, weak_labels_df, validation_df)
        """
        click.echo("Loading artifacts...")

        artifacts = load_artifacts(artifacts_dir)

        click.echo(f"✓ Loaded {len(artifacts['features'])} feature records")
        click.echo(f"✓ Loaded {len(artifacts['embeddings'])} embeddings")
        click.echo(f"✓ Loaded {len(artifacts['weak_labels'])} weak labels")
        click.echo(f"✓ Loaded {len(artifacts['validation'])} validation records")

        return (
            artifacts["features"],
            artifacts["embeddings"],
            artifacts["weak_labels"],
            artifacts["validation"],
        )

    def prepare_features(
        self,
        features_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Prepare features using ranker.

        Args:
            features_df: Features DataFrame
            embeddings_df: Embeddings DataFrame
            validation_df: Validation DataFrame

        Returns:
            Tuple of (X, groups)
        """
        click.echo("Preparing features...")

        X, groups = self.ranker.prepare_features(
            features_df, embeddings_df, validation_df
        )

        click.echo(f"✓ Prepared {X.shape[0]} samples with {X.shape[1]} features")
        click.echo(f"✓ Created {len(groups)} groups")
        click.echo(f"✓ Feature names: {self.ranker.feature_names}")

        return X, groups

    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: List[int],
        features_df: pd.DataFrame,
    ) -> Dict:
        """Split data into train/val/test sets.

        Args:
            X: Feature matrix
            y: Labels
            groups: Group sizes
            features_df: Original features (for group column)

        Returns:
            Dictionary with train/val/test splits
        """
        click.echo("Splitting data...")

        training_cfg = self.training_config.get_training_params()
        random_seed = training_cfg["random_seed"]

        # Create group IDs for splitting
        group_col = self.ranker.group_by
        if group_col and group_col in features_df.columns:
            group_ids = features_df[group_col].fillna("none").values
        else:
            # No grouping - each sample is its own group
            group_ids = np.arange(len(X))

        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=training_cfg["test_split"],
            random_state=random_seed,
            stratify=None,  # Can't stratify with groups
        )

        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=training_cfg["val_split"] / (1 - training_cfg["test_split"]),
            random_state=random_seed,
        )

        # Extract splits
        splits = {
            "train": {
                "X": X.iloc[train_idx],
                "y": y[train_idx],
                "groups": self._compute_groups(group_ids[train_idx]),
            },
            "val": {
                "X": X.iloc[val_idx],
                "y": y[val_idx],
                "groups": self._compute_groups(group_ids[val_idx]),
            },
            "test": {
                "X": X.iloc[test_idx],
                "y": y[test_idx],
                "groups": self._compute_groups(group_ids[test_idx]),
            },
        }

        click.echo(f"✓ Train: {len(train_idx)} samples, {len(splits['train']['groups'])} groups")
        click.echo(f"✓ Val:   {len(val_idx)} samples, {len(splits['val']['groups'])} groups")
        click.echo(f"✓ Test:  {len(test_idx)} samples, {len(splits['test']['groups'])} groups")

        return splits

    @staticmethod
    def _compute_groups(group_ids: np.ndarray) -> List[int]:
        """Compute group sizes from group IDs.

        Args:
            group_ids: Array of group identifiers

        Returns:
            List of group sizes
        """
        unique_groups, counts = np.unique(group_ids, return_counts=True)
        return counts.tolist()

    def train(
        self,
        splits: Dict,
        save_path: Optional[str] = None,
    ) -> Dict:
        """Train LightGBM ranker.

        Args:
            splits: Dictionary with train/val/test splits
            save_path: Optional path to save trained model

        Returns:
            Training metrics
        """
        click.echo("\n" + "=" * 60)
        click.echo("TRAINING LIGHTGBM LAMBDAMART RANKER")
        click.echo("=" * 60)

        # Prepare validation set
        eval_set = (
            splits["val"]["X"],
            splits["val"]["y"],
            splits["val"]["groups"],
        )

        # Train model
        click.echo("\nTraining model...")
        self.train_metrics = self.ranker.train(
            splits["train"]["X"],
            splits["train"]["y"],
            splits["train"]["groups"],
            eval_set=eval_set,
        )

        click.echo("\n✓ Training complete!")

        # Save model
        if save_path is None:
            model_dir = Path(self.training_config.get_training_params()["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(model_dir / "ltr_model.txt")

        self.ranker.save(save_path)
        click.echo(f"✓ Model saved to: {save_path}")

        return self.train_metrics

    def evaluate(
        self,
        splits: Dict,
        dataset_name: str = "test",
    ) -> Dict:
        """Evaluate trained model.

        Args:
            splits: Data splits
            dataset_name: Which split to evaluate ("train", "val", "test")

        Returns:
            Evaluation metrics
        """
        click.echo(f"\nEvaluating on {dataset_name} set...")

        split = splits[dataset_name]

        # Predict scores
        y_pred = self.ranker.predict(split["X"])

        # Compute metrics
        metrics = compute_ranking_metrics(
            split["y"],
            y_pred,
            split["groups"],
        )

        click.echo(f"✓ {dataset_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            click.echo(f"  {metric_name}: {value:.4f}")

        return metrics

    def save_report(
        self,
        splits: Dict,
        output_path: Optional[str] = None,
    ):
        """Save training report with all metrics and analysis.

        Args:
            splits: Data splits
            output_path: Optional output path
        """
        if output_path is None:
            reports_dir = Path(self.training_config.get_training_params()["results_dir"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(reports_dir / "training_report.json")

        # Evaluate all splits
        train_metrics = self.evaluate(splits, "train")
        val_metrics = self.evaluate(splits, "val")
        test_metrics = self.evaluate(splits, "test")

        # Compile report
        report = {
            "config": self.training_config.to_dict(),
            "data": {
                "train_size": len(splits["train"]["X"]),
                "val_size": len(splits["val"]["X"]),
                "test_size": len(splits["test"]["X"]),
                "n_features": splits["train"]["X"].shape[1],
                "feature_names": self.ranker.feature_names,
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "model": {
                "type": "LightGBM LambdaMART",
                "n_estimators": self.ranker.n_estimators,
                "num_leaves": self.ranker.num_leaves,
                "learning_rate": self.ranker.learning_rate,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        click.echo(f"\n✓ Training report saved to: {output_path}")

        return report


@click.command()
@click.option(
    "--artifacts-dir",
    default="./artifacts",
    help="Path to artifacts directory",
)
@click.option(
    "--config",
    default=None,
    help="Path to config file",
)
@click.option(
    "--output-dir",
    default="./training/model",
    help="Output directory for trained model",
)
def main(artifacts_dir, config, output_dir):
    """Train LightGBM LambdaMART ranker using weak labels."""

    click.echo("\n" + "=" * 60)
    click.echo("JIRA RANKER TRAINING")
    click.echo("=" * 60 + "\n")

    # Initialize trainer
    trainer = RankerTrainer(config)

    # Load data
    features_df, embeddings_df, weak_labels_df, validation_df = trainer.load_data(
        artifacts_dir
    )

    # Prepare features
    X, groups = trainer.prepare_features(features_df, embeddings_df, validation_df)

    # Get labels
    y_continuous = weak_labels_df["relevance_score"].values

    # Convert continuous scores to integer grades for LambdaRank
    # LambdaRank requires integer labels (0, 1, 2, 3) that map to label_gain
    click.echo("\nConverting continuous weak labels to discrete grades...")
    click.echo(f"Original range: [{y_continuous.min():.3f}, {y_continuous.max():.3f}]")

    # Use quartile-based binning for balanced distribution
    q1 = np.percentile(y_continuous, 25)
    q2 = np.percentile(y_continuous, 50)
    q3 = np.percentile(y_continuous, 75)

    y = np.zeros(len(y_continuous), dtype=np.int32)
    y[y_continuous <= q1] = 0  # Low relevance
    y[(y_continuous > q1) & (y_continuous <= q2)] = 1  # Medium relevance
    y[(y_continuous > q2) & (y_continuous <= q3)] = 2  # High relevance
    y[y_continuous > q3] = 3  # Very high relevance

    # Show distribution
    unique, counts = np.unique(y, return_counts=True)
    click.echo(f"Grade distribution: {dict(zip(unique, counts))}")
    click.echo(f"Quartile thresholds: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}\n")

    # Split data
    splits = trainer.split_data(X, y, groups, features_df)

    # Train model
    model_path = Path(output_dir) / "ltr_model.txt"
    trainer.train(splits, save_path=str(model_path))

    # Evaluate and save report
    trainer.save_report(splits)

    click.echo("\n" + "=" * 60)
    click.echo("✓ TRAINING COMPLETE!")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
