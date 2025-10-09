"""Training-specific configuration for LightGBM LambdaMART ranker."""

from pathlib import Path
from typing import Dict, List, Optional

import yaml


class TrainingConfig:
    """Configuration for training LightGBM ranker."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize training config.

        Args:
            config_path: Optional path to YAML config file
        """
        # Load base config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"

        with open(config_path) as f:
            self.base_config = yaml.safe_load(f)

        # Training-specific settings
        self.training_config = {
            # Data splits
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "random_seed": 42,

            # Cross-validation
            "cv_folds": 5,
            "cv_strategy": "group",  # group by epic/sprint

            # Model selection
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "optuna",  # optuna | grid | random
                "n_trials": 50,
                "timeout": 3600,  # 1 hour
            },

            # Training settings
            "early_stopping_rounds": 50,
            "eval_metric": "ndcg@10",
            "verbose_eval": 10,

            # Weak label filtering
            "weak_label_confidence_threshold": 0.0,  # Filter low-confidence labels
            "label_smoothing": 0.0,  # Optional label smoothing

            # Feature selection
            "feature_importance_threshold": 0.01,  # Remove low-importance features
            "max_features": None,  # Limit number of features

            # Experiment tracking
            "mlflow": {
                "enabled": False,
                "tracking_uri": "./training/experiments/mlruns",
                "experiment_name": "jira_ranker",
            },

            # Model outputs
            "model_dir": "./training/model",
            "checkpoint_dir": "./training/model/checkpoints",
            "results_dir": "./training/reports",

            # Comparison
            "baseline_heuristic": {
                "priority_weight": 0.4,
                "status_weight": 0.3,
                "hygiene_weight": 0.3,
            },
        }

    def get_ranker_params(self) -> Dict:
        """Get LightGBM ranker parameters from base config."""
        return self.base_config.get("ranker", {})

    def get_training_params(self) -> Dict:
        """Get training-specific parameters."""
        return self.training_config

    def get_artifacts_config(self) -> Dict:
        """Get artifact management configuration."""
        return self.base_config.get("artifacts", {})

    def to_dict(self) -> Dict:
        """Export full config as dictionary."""
        return {
            "base": self.base_config,
            "training": self.training_config,
        }


def load_training_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load training configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(config_path)
