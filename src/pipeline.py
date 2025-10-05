"""Main pipeline orchestrator - coordinates all stages."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .core import (
    create_embedder,
    create_feature_extractor,
    create_indexer,
    create_preprocessor,
    create_ranker,
    create_reranker,
    create_validator,
    create_weak_labeler,
)
from .utils import ArtifactManager, StructuredLogger, benchmark_stage, create_logger


class JIRAPipeline:
    """End-to-end pipeline: JIRA → Validation → Features → Embeddings → Ranking."""

    def __init__(self, config: Dict[str, any], logger: Optional[StructuredLogger] = None):
        """Initialize pipeline.

        Args:
            config: Configuration dictionary
            logger: Optional logger (will create if not provided)
        """
        self.config = config
        self.logger = logger or create_logger(config, name="pipeline")

        # Initialize components
        self.validator = create_validator(config)
        self.feature_extractor = create_feature_extractor(config)
        self.weak_labeler = create_weak_labeler(config)
        self.preprocessor = create_preprocessor(config)
        self.embedder = create_embedder(config)
        self.indexer = create_indexer(config)
        self.ranker = create_ranker(config)
        self.reranker = create_reranker(config)

        # Artifact manager
        artifacts_cfg = config.get("artifacts", {})
        self.artifacts = ArtifactManager(
            base_dir=artifacts_cfg.get("base_dir", "./artifacts"),
            version_artifacts=artifacts_cfg.get("version_artifacts", True),
        )

        self.logger.info("Pipeline initialized", components=self._get_component_status())

    def run(
        self,
        input_path: str,
        input_format: str = "auto",
        skip_validation: bool = False,
        skip_training: bool = True,
    ) -> pd.DataFrame:
        """Run full pipeline.

        Args:
            input_path: Path to JIRA export (XML or CSV)
            input_format: Format hint ('auto', 'xml', 'csv')
            skip_validation: Skip validation stage (use cached)
            skip_training: Skip model training (use existing model)

        Returns:
            Final ranked backlog DataFrame
        """
        self.logger.info("Starting pipeline", input_path=input_path, format=input_format)

        # Stage 1: Validation
        if not skip_validation:
            report_df, summary = self._run_validation(input_path, input_format)
        else:
            self.logger.info("Skipping validation (using cached)")
            report_df = self._load_cached("validation", "backbone_report")
            summary = None

        # Stage 2: Feature Extraction
        features_df = self._run_feature_extraction(input_path, input_format)

        # Stage 3: Weak Labeling (if enabled)
        weak_labels_df = None
        weak_labels_cfg = self.config.get("ranker", {}).get("weak_labels", {})
        if weak_labels_cfg.get("enabled", False):
            weak_labels_df = self._run_weak_labeling(features_df, report_df)

        # Stage 4: Preprocessing/Augmentation (if enabled)
        preprocessing_cfg = self.config.get("preprocessing", {})
        if preprocessing_cfg.get("enabled", False):
            features_df, weak_labels_df = self._run_preprocessing(features_df, weak_labels_df)

        # Stage 5: Embeddings
        embeddings_df = self._run_embeddings(features_df)

        # Stage 6: FAISS Indexing (optional, for retrieval)
        self._run_indexing(embeddings_df)

        # Stage 7: Ranking
        if not skip_training:
            # In production, you'd have labels here
            # For now, we'll use weak labels or skip training
            self.logger.warning(
                "Training skipped - requires labeled data. Using inference mode."
            )

        ranked_df = self._run_ranking(features_df, embeddings_df, report_df)

        # Stage 6: Re-ranking (optional)
        if self.reranker:
            ranked_df = self._run_reranking(ranked_df, features_df)

        # Stage 7: Apply guardrails
        final_df = self._apply_guardrails(ranked_df)

        # Save final output
        output_path = self.artifacts.save_dataframe(
            final_df,
            category="backlogs",
            name="clean_backlog",
            format="csv",
            metadata={"input_path": input_path, "format": input_format},
        )

        self.logger.info("Pipeline completed", output_path=str(output_path))
        self.logger.save_benchmarks(suffix="_pipeline")

        return final_df

    @benchmark_stage(stage_name="validation")
    def _run_validation(self, input_path: str, input_format: str):
        """Run validation stage."""
        self.logger.info("Running validation stage")

        report_df, summary = self.validator.validate_file(input_path, fmt=input_format)

        # Save artifacts
        self.artifacts.save_dataframe(
            report_df,
            category="validation",
            name="backbone_report",
            format="csv",
        )

        self.artifacts.save_json(
            summary,
            category="validation",
            name="backbone_summary",
        )

        # Check for critical errors
        if summary["errors"]["missing_required"] > 0:
            self.logger.error(
                "Validation failed - missing required fields",
                count=summary["errors"]["missing_required"],
            )
            raise ValueError("Validation failed. See backbone_report.csv for details.")

        self.logger.info("Validation completed", summary=summary)
        return report_df, summary

    @benchmark_stage(stage_name="feature_extraction")
    def _run_feature_extraction(self, input_path: str, input_format: str):
        """Run feature extraction stage."""
        self.logger.info("Running feature extraction stage")

        # Load rows
        rows = self.validator._load_rows(input_path, input_format)

        # Extract features
        features_df = self.feature_extractor.extract(rows)

        # Save artifacts
        self.artifacts.save_dataframe(
            features_df,
            category="features",
            name="variability_features",
            format="parquet",
        )

        self.logger.info("Feature extraction completed", features_count=len(features_df))
        return features_df

    @benchmark_stage(stage_name="weak_labeling")
    def _run_weak_labeling(self, features_df: pd.DataFrame, validation_df: pd.DataFrame):
        """Run weak labeling stage."""
        self.logger.info("Running weak labeling stage")

        weak_labels_df = self.weak_labeler.generate_labels(features_df, validation_df)

        # Save artifacts
        self.artifacts.save_dataframe(
            weak_labels_df,
            category="labels",
            name="weak_labels",
            format="csv",
        )

        self.logger.info("Weak labeling completed", labels_count=len(weak_labels_df))
        return weak_labels_df

    @benchmark_stage(stage_name="preprocessing")
    def _run_preprocessing(
        self, features_df: pd.DataFrame, labels_df: Optional[pd.DataFrame]
    ):
        """Run preprocessing/augmentation stage."""
        self.logger.info("Running preprocessing/augmentation stage")

        augmented_features, augmented_labels = self.preprocessor.augment_dataset(
            features_df, labels_df
        )

        self.logger.info(
            "Preprocessing completed",
            original_count=len(features_df),
            augmented_count=len(augmented_features),
        )
        return augmented_features, augmented_labels

    @benchmark_stage(stage_name="embeddings")
    def _run_embeddings(self, features_df: pd.DataFrame):
        """Run embeddings stage."""
        self.logger.info("Running embeddings stage")

        embeddings_df = self.embedder.embed_features(features_df)

        # Save artifacts
        self.artifacts.save_dataframe(
            embeddings_df,
            category="embeddings",
            name="embeddings",
            format="parquet",
        )

        self.logger.info("Embeddings completed", embeddings_count=len(embeddings_df))
        return embeddings_df

    @benchmark_stage(stage_name="indexing")
    def _run_indexing(self, embeddings_df: pd.DataFrame):
        """Run FAISS indexing stage."""
        self.logger.info("Running FAISS indexing stage")

        self.indexer.build(embeddings_df)

        # Save index
        index_path = (
            Path(self.config["artifacts"]["base_dir"]) / "indices" / "faiss_index.ivf"
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.indexer.save(str(index_path))

        stats = self.indexer.get_index_stats()
        self.logger.info("FAISS indexing completed", stats=stats)

    @benchmark_stage(stage_name="ranking")
    def _run_ranking(
        self,
        features_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ):
        """Run ranking stage."""
        self.logger.info("Running ranking stage")

        # For inference without training, use simple heuristic ranking
        # In production, you'd load a trained model here

        # Combine features
        df = features_df.merge(embeddings_df, on="key", how="inner")
        df = df.merge(
            validation_df[["key", "required_ok", "dates_ok"]], on="key", how="left"
        )

        # Simple heuristic scoring (placeholder)
        # Priority: high=3, medium=2, low=1
        priority_map = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        df["priority_score"] = df["priority"].map(priority_map).fillna(0)

        # Status: in_progress=3, todo=2, blocked=1
        status_map = {"in progress": 3, "to do": 2, "blocked": 1, "unknown": 0}
        df["status_score"] = df["status"].map(status_map).fillna(0)

        # Hygiene bonus
        df["hygiene_score"] = df["required_ok"].astype(int) + df["dates_ok"].astype(int)

        # Combined score
        df["score"] = (
            0.4 * df["priority_score"]
            + 0.3 * df["status_score"]
            + 0.3 * df["hygiene_score"]
        )

        # Rank
        df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)

        df = df.sort_values("rank").reset_index(drop=True)

        self.logger.info("Ranking completed", ranked_count=len(df))
        return df

    @benchmark_stage(stage_name="reranking")
    def _run_reranking(self, ranked_df: pd.DataFrame, features_df: pd.DataFrame):
        """Run re-ranking stage."""
        self.logger.info("Running re-ranking stage")

        reranked_df = self.reranker.rerank(ranked_df, features_df)

        self.logger.info("Re-ranking completed")
        return reranked_df

    def _apply_guardrails(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rule guardrails."""
        guardrails_cfg = self.config.get("guardrails", {})

        if not guardrails_cfg.get("enabled", False):
            return ranked_df

        self.logger.info("Applying guardrails")

        # Placeholder: implement specific rules from config
        # For now, just return unchanged

        return ranked_df

    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of pipeline components."""
        return {
            "validator": self.validator is not None,
            "feature_extractor": self.feature_extractor is not None,
            "embedder": self.embedder is not None,
            "indexer": self.indexer is not None,
            "ranker": self.ranker is not None,
            "reranker": self.reranker is not None,
        }

    def _load_cached(self, category: str, name: str) -> pd.DataFrame:
        """Load cached artifact."""
        latest = self.artifacts.get_latest(category, name, format="csv")
        if latest:
            return pd.read_csv(latest)

        latest_parquet = self.artifacts.get_latest(category, name, format="parquet")
        if latest_parquet:
            return pd.read_parquet(latest_parquet)

        raise FileNotFoundError(f"No cached artifact found: {category}/{name}")
