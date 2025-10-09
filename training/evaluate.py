"""Evaluation module for trained LightGBM ranker."""

import json
from pathlib import Path
from typing import Dict, Optional

import click
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ranker import LambdaMARTRanker
from training.utils import load_artifacts, compute_ranking_metrics


class RankerEvaluator:
    """Evaluates trained LightGBM ranker."""

    def __init__(self, model_path: str):
        """Initialize evaluator.

        Args:
            model_path: Path to trained model
        """
        self.ranker = LambdaMARTRanker()
        self.ranker.load(model_path)
        click.echo(f"✓ Loaded model from: {model_path}")

    def evaluate_on_dataset(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """Evaluate model on a dataset.

        Args:
            dataset_path: Path to dataset CSV
            output_path: Optional output path for results

        Returns:
            Evaluation metrics
        """
        click.echo(f"Evaluating on: {dataset_path}")

        # Load dataset
        df = pd.read_csv(dataset_path)

        # TODO: Implement evaluation logic
        # This would require running feature extraction, etc.

        raise NotImplementedError("Dataset evaluation not yet implemented")

    def evaluate_on_artifacts(
        self,
        artifacts_dir: str = "./artifacts",
        output_path: Optional[str] = None,
    ) -> Dict:
        """Evaluate model using preprocessed artifacts.

        Args:
            artifacts_dir: Path to artifacts directory
            output_path: Optional output path for results

        Returns:
            Evaluation metrics
        """
        click.echo("Loading artifacts...")

        # Load artifacts
        artifacts = load_artifacts(artifacts_dir)

        # Prepare features
        X, groups = self.ranker.prepare_features(
            artifacts["features"],
            artifacts["embeddings"],
            artifacts["validation"],
        )

        # Get labels
        y_true = artifacts["weak_labels"]["relevance_score"].values

        # Predict
        click.echo("Generating predictions...")
        y_pred = self.ranker.predict(X)

        # Compute metrics
        click.echo("Computing metrics...")
        metrics = compute_ranking_metrics(y_true, y_pred, groups)

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("EVALUATION RESULTS")
        click.echo("=" * 60)
        for metric_name, value in metrics.items():
            click.echo(f"  {metric_name}: {value:.4f}")
        click.echo("=" * 60)

        # Save results
        if output_path:
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)
            click.echo(f"\n✓ Results saved to: {output_path}")

        return metrics

    def predict_and_rank(
        self,
        artifacts_dir: str = "./artifacts",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate predictions and create ranked output.

        Args:
            artifacts_dir: Path to artifacts directory
            output_path: Optional output path for ranked CSV

        Returns:
            Ranked DataFrame
        """
        click.echo("Generating ranked output...")

        # Load artifacts
        artifacts = load_artifacts(artifacts_dir)

        # Prepare features
        X, groups = self.ranker.prepare_features(
            artifacts["features"],
            artifacts["embeddings"],
            artifacts["validation"],
        )

        # Predict scores
        scores = self.ranker.predict(X)

        # Create ranked output
        result_df = artifacts["features"][["key"]].copy()
        result_df["ml_score"] = scores
        result_df["ml_rank"] = result_df["ml_score"].rank(ascending=False, method="first").astype(int)

        # Add weak label for comparison
        result_df = result_df.merge(
            artifacts["weak_labels"][["key", "relevance_score"]],
            on="key",
            how="left",
        )

        # Sort by ML rank
        result_df = result_df.sort_values("ml_rank").reset_index(drop=True)

        click.echo(f"✓ Generated {len(result_df)} ranked predictions")

        # Save
        if output_path:
            result_df.to_csv(output_path, index=False)
            click.echo(f"✓ Ranked output saved to: {output_path}")

        return result_df


@click.group()
def cli():
    """Evaluation tools for trained LightGBM ranker."""
    pass


@cli.command()
@click.option(
    "--model",
    required=True,
    help="Path to trained model file",
)
@click.option(
    "--artifacts-dir",
    default="./artifacts",
    help="Path to artifacts directory",
)
@click.option(
    "--output",
    default=None,
    help="Output path for results",
)
def evaluate(model, artifacts_dir, output):
    """Evaluate trained model on artifacts."""
    evaluator = RankerEvaluator(model)
    evaluator.evaluate_on_artifacts(artifacts_dir, output)


@click.command()
@click.option(
    "--model",
    required=True,
    help="Path to trained model file",
)
@click.option(
    "--artifacts-dir",
    default="./artifacts",
    help="Path to artifacts directory",
)
@click.option(
    "--output",
    default="./training/reports/ranked_output.csv",
    help="Output path for ranked CSV",
)
def rank(model, artifacts_dir, output):
    """Generate ranked output using trained model."""
    evaluator = RankerEvaluator(model)
    evaluator.predict_and_rank(artifacts_dir, output)


if __name__ == "__main__":
    cli()
