"""LightGBM LambdaMART ranker for learning-to-rank."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class LambdaMARTRanker:
    """LightGBM-based learning-to-rank model."""

    def __init__(
        self,
        objective: str = "lambdarank",
        metric: str = "ndcg",
        label_gain: List[int] = None,
        num_leaves: int = 31,
        learning_rate: float = 0.06,
        n_estimators: int = 500,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        eval_at: List[int] = None,
        group_by: Optional[str] = "epic",
    ):
        """Initialize LambdaMART ranker.

        Args:
            objective: LightGBM objective ('lambdarank')
            metric: Evaluation metric ('ndcg')
            label_gain: Gain values for relevance levels
            num_leaves: Max leaves per tree
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            min_data_in_leaf: Minimum samples per leaf
            feature_fraction: Feature sampling fraction
            bagging_fraction: Data sampling fraction
            bagging_freq: Bagging frequency
            eval_at: NDCG@K values to track
            group_by: Column to group by ('epic', 'sprint', or None)
        """
        self.objective = objective
        self.metric = metric
        self.label_gain = label_gain or [0, 1, 3, 7]
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.eval_at = eval_at or [10, 20, 50]
        self.group_by = group_by

        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.encoders: Dict[str, any] = {}

    def prepare_features(
        self,
        features_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Prepare features for ranking.

        Args:
            features_df: Variability features
            embeddings_df: Embeddings with similarity scores
            validation_df: Validation flags

        Returns:
            Tuple of (feature_matrix, group_sizes)
        """
        # Merge all sources
        df = features_df.merge(embeddings_df, on="key", how="inner")
        df = df.merge(validation_df[["key", "required_ok", "dates_ok"]], on="key", how="left")

        # Encode categorical features
        categorical_cols = ["type", "status", "priority"]
        for col in categorical_cols:
            if col not in df.columns:
                continue

            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f"{col}_encoded"] = self.encoders[col].fit_transform(df[col].fillna("unknown"))
            else:
                df[f"{col}_encoded"] = self.encoders[col].transform(df[col].fillna("unknown"))

        # Select features
        feature_cols = [
            # Text lengths
            "summary_len",
            "description_len",
            # Counts
            "label_count",
            "component_count",
            "customfield_count",
            "link_count",
            # Dirty flags
            "flag_missing_ac",
            "assignee_empty",
            "storypoints_empty",
            # Categorical (encoded)
            "type_encoded",
            "status_encoded",
            "priority_encoded",
            # Hygiene flags (from validation)
            "required_ok",
            "dates_ok",
        ]

        # Add embedding if available (as single feature or similarity score)
        if "embedding" in df.columns:
            # For simplicity, use embedding norm as a feature
            df["embedding_norm"] = df["embedding"].apply(lambda x: np.linalg.norm(x))
            feature_cols.append("embedding_norm")

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        self.feature_names = available_cols

        # Convert boolean to int
        for col in available_cols:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        X = df[available_cols].fillna(0).astype(float)

        # Create group sizes for ranking (group by epic/sprint)
        group_sizes = []
        if self.group_by and self.group_by in df.columns:
            df["_group"] = df[self.group_by].fillna("none")
            for group in df["_group"].unique():
                group_size = (df["_group"] == group).sum()
                group_sizes.append(group_size)
        else:
            # Single group (all issues)
            group_sizes = [len(df)]

        return X, group_sizes

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group_sizes: List[int],
        eval_set: Optional[Tuple] = None,
    ) -> Dict[str, float]:
        """Train LambdaMART model.

        Args:
            X: Feature matrix
            y: Labels (relevance scores)
            group_sizes: Size of each group
            eval_set: Optional (X_val, y_val, group_val) for validation

        Returns:
            Training metrics
        """
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            group=group_sizes,
            feature_name=self.feature_names,
        )

        # Parameters
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "label_gain": self.label_gain,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_data_in_leaf": self.min_data_in_leaf,
            "verbose": -1,
        }

        # Add NDCG@K metrics
        eval_at_str = ",".join(map(str, self.eval_at))
        params["ndcg_eval_at"] = eval_at_str

        # Validation set
        valid_sets = [train_data]
        if eval_set:
            X_val, y_val, group_val = eval_set
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=group_val,
                reference=train_data,
            )
            valid_sets.append(valid_data)

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
        )

        # Get metrics
        metrics = {}
        if hasattr(self.model, "best_score"):
            metrics = self.model.best_score

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.

        Args:
            X: Feature matrix

        Returns:
            Ranking scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def rank(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ranked backlog.

        Args:
            features_df: Features DataFrame with all necessary columns

        Returns:
            DataFrame with ranking scores and positions
        """
        # Prepare features (simplified, assumes already prepared)
        X = features_df[self.feature_names].fillna(0).astype(float)

        # Predict scores
        scores = self.predict(X)

        # Add scores and rank
        result = features_df.copy()
        result["score"] = scores
        result["rank"] = result["score"].rank(ascending=False, method="first").astype(int)

        # Sort by rank
        result = result.sort_values("rank").reset_index(drop=True)

        return result

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.save_model(filepath)

        # Save feature names and encoders
        import json
        import pickle

        meta_path = Path(filepath).with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "group_by": self.group_by,
                },
                f,
                indent=2,
            )

        encoder_path = Path(filepath).with_suffix(".encoders.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(self.encoders, f)

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to model file
        """
        self.model = lgb.Booster(model_file=filepath)

        # Load metadata
        import json
        import pickle

        meta_path = Path(filepath).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.feature_names = meta["feature_names"]
                self.group_by = meta.get("group_by")

        encoder_path = Path(filepath).with_suffix(".encoders.pkl")
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                self.encoders = pickle.load(f)


def create_ranker(config: Dict[str, any]) -> LambdaMARTRanker:
    """Factory function to create ranker from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LambdaMARTRanker instance
    """
    ranker_cfg = config.get("ranker", {})

    return LambdaMARTRanker(
        objective=ranker_cfg.get("objective", "lambdarank"),
        metric=ranker_cfg.get("metric", "ndcg"),
        label_gain=ranker_cfg.get("label_gain", [0, 1, 3, 7]),
        num_leaves=ranker_cfg.get("num_leaves", 31),
        learning_rate=ranker_cfg.get("learning_rate", 0.06),
        n_estimators=ranker_cfg.get("n_estimators", 500),
        min_data_in_leaf=ranker_cfg.get("min_data_in_leaf", 20),
        feature_fraction=ranker_cfg.get("feature_fraction", 0.8),
        bagging_fraction=ranker_cfg.get("bagging_fraction", 0.8),
        bagging_freq=ranker_cfg.get("bagging_freq", 5),
        eval_at=ranker_cfg.get("eval_at", [10, 20, 50]),
        group_by=ranker_cfg.get("group_by", "epic"),
    )
