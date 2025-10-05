"""Weak label generation for LambdaMART training when ground truth is unavailable.

This module generates synthetic relevance scores based on business rules and metadata,
allowing the ranking model to learn patterns from heuristic priorities.
"""

import os
from typing import Dict, List, Optional

import pandas as pd


class WeakLabeler:
    """Generates weak labels from JIRA metadata for ranking model training."""

    def __init__(
        self,
        priority_weight: float = 0.4,
        status_weight: float = 0.3,
        hygiene_weight: float = 0.3,
        priority_scores: Optional[Dict[str, float]] = None,
        status_scores: Optional[Dict[str, float]] = None,
        require_human_approval: bool = False,
    ):
        """Initialize weak labeler.

        Args:
            priority_weight: Weight for priority contribution (0-1)
            status_weight: Weight for status contribution (0-1)
            hygiene_weight: Weight for hygiene/completeness contribution (0-1)
            priority_scores: Custom priority to score mapping
            status_scores: Custom status to score mapping
            require_human_approval: If True, prompts for human review before finalizing
        """
        self.priority_weight = priority_weight
        self.status_weight = status_weight
        self.hygiene_weight = hygiene_weight
        self.require_human_approval = require_human_approval

        # Default priority scoring (aligned with JIRA standard priorities)
        self.priority_scores = priority_scores or {
            "critical": 4.0,
            "highest": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0,
            "lowest": 0.5,
            "trivial": 0.5,
        }

        # Default status scoring (work in progress > ready > backlog > blocked)
        self.status_scores = status_scores or {
            "in progress": 4.0,
            "in development": 4.0,
            "to do": 3.0,
            "ready": 3.0,
            "selected for development": 3.0,
            "backlog": 2.0,
            "new": 2.0,
            "blocked": 1.0,
            "on hold": 1.0,
            "done": 0.0,
            "closed": 0.0,
            "resolved": 0.0,
            "cancelled": 0.0,
        }

    def generate_labels(
        self,
        features_df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate weak labels from features and validation data.

        Args:
            features_df: DataFrame with variability features (type, status, priority, etc.)
            validation_df: DataFrame with validation flags (required_ok, dates_ok, etc.)

        Returns:
            DataFrame with 'key' and 'relevance_score' columns
        """
        # Merge features and validation
        df = features_df.merge(
            validation_df[["key", "required_ok", "dates_ok", "key_format_ok"]],
            on="key",
            how="inner",
        )

        # Calculate component scores
        df["priority_score"] = df["priority"].str.lower().map(self.priority_scores).fillna(1.0)
        df["status_score"] = df["status"].str.lower().map(self.status_scores).fillna(1.0)
        df["hygiene_score"] = self._calculate_hygiene_score(df)

        # Normalize scores to 0-1 range
        df["priority_score_norm"] = self._normalize(df["priority_score"])
        df["status_score_norm"] = self._normalize(df["status_score"])
        df["hygiene_score_norm"] = self._normalize(df["hygiene_score"])

        # Weighted combination
        df["relevance_score"] = (
            self.priority_weight * df["priority_score_norm"]
            + self.status_weight * df["status_score_norm"]
            + self.hygiene_weight * df["hygiene_score_norm"]
        )

        # Scale to 0-4 range for LambdaMART label_gain compatibility
        df["relevance_score"] = (df["relevance_score"] * 4).round(2)

        # Apply business rule adjustments
        df = self._apply_business_rules(df)

        # Human approval if required
        if self.require_human_approval:
            df = self._request_human_approval(df)

        # Return only key and score
        result = df[["key", "relevance_score"]].copy()

        return result

    def _calculate_hygiene_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate hygiene/completeness score from validation flags and features.

        Args:
            df: Combined features and validation DataFrame

        Returns:
            Hygiene score series
        """
        score = pd.Series(0.0, index=df.index)

        # Required fields present
        score += df["required_ok"].astype(int) * 1.0

        # Dates valid
        score += df["dates_ok"].astype(int) * 1.0

        # Key format valid
        score += df["key_format_ok"].astype(int) * 0.5

        # Has assignee (from dirty flags)
        score += (1 - df["assignee_empty"]) * 1.0

        # Has story points or estimates (inverse of storypoints_empty)
        score += (1 - df["storypoints_empty"]) * 0.5

        # Has acceptance criteria (inverse of flag_missing_ac)
        score += (1 - df["flag_missing_ac"]) * 1.0

        # Has description
        score += (df["description_len"] > 20).astype(int) * 0.5

        # Has labels/components (shows categorization effort)
        score += (df["label_count"] > 0).astype(int) * 0.25
        score += (df["component_count"] > 0).astype(int) * 0.25

        return score

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Normalize series to 0-1 range using min-max scaling.

        Args:
            series: Input series

        Returns:
            Normalized series
        """
        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            return pd.Series(0.5, index=series.index)

        return (series - min_val) / (max_val - min_val)

    def _apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business logic adjustments to scores.

        Args:
            df: DataFrame with relevance_score

        Returns:
            Adjusted DataFrame
        """
        # Critical bugs get minimum score boost
        mask_critical_bug = (df["type"].str.lower() == "bug") & (
            df["priority"].str.lower().isin(["critical", "highest", "high"])
        )
        df.loc[mask_critical_bug, "relevance_score"] = df.loc[mask_critical_bug, "relevance_score"] + 0.5

        # Blocked/On Hold tickets get capped at lower percentile
        mask_blocked = df["status"].str.lower().isin(["blocked", "on hold"])
        df.loc[mask_blocked, "relevance_score"] = df.loc[mask_blocked, "relevance_score"] * 0.3

        # Very incomplete tickets (multiple dirty flags) get penalized
        dirty_count = (
            df["flag_missing_ac"] + df["assignee_empty"] + df["storypoints_empty"]
        )
        mask_very_dirty = dirty_count >= 2
        df.loc[mask_very_dirty, "relevance_score"] = df.loc[mask_very_dirty, "relevance_score"] * 0.7

        # Ensure scores stay within 0-4 range
        df["relevance_score"] = df["relevance_score"].clip(0, 4)

        return df

    def _request_human_approval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Request human approval for generated labels (if environment variable set).

        Args:
            df: DataFrame with generated labels

        Returns:
            Approved DataFrame
        """
        # Check environment variable
        approval_mode = os.getenv("WEAK_LABEL_APPROVAL", "auto")

        if approval_mode.lower() == "manual":
            print("\n" + "=" * 80)
            print("WEAK LABEL GENERATION - HUMAN REVIEW REQUIRED")
            print("=" * 80)
            print(f"\nGenerated {len(df)} weak labels")
            print("\nScore distribution:")
            print(df["relevance_score"].describe())
            print("\nTop 10 highest scored issues:")
            print(df.nlargest(10, "relevance_score")[["key", "type", "status", "priority", "relevance_score"]])
            print("\nBottom 10 lowest scored issues:")
            print(df.nsmallest(10, "relevance_score")[["key", "type", "status", "priority", "relevance_score"]])

            response = input("\nApprove these labels? (yes/no/adjust): ").lower()

            if response == "no":
                print("Labels rejected. Exiting.")
                raise ValueError("Human rejected weak labels")
            elif response == "adjust":
                print("\nAdjustment options:")
                print("1. Increase priority weight")
                print("2. Increase status weight")
                print("3. Increase hygiene weight")
                choice = input("Select option (1-3): ")

                if choice == "1":
                    self.priority_weight = min(1.0, self.priority_weight + 0.1)
                elif choice == "2":
                    self.status_weight = min(1.0, self.status_weight + 0.1)
                elif choice == "3":
                    self.hygiene_weight = min(1.0, self.hygiene_weight + 0.1)

                # Regenerate with adjusted weights
                return self.generate_labels(
                    df[features_df.columns], df[validation_df.columns]
                )

        return df

    def export_labels(self, labels_df: pd.DataFrame, output_path: str) -> None:
        """Export weak labels to file.

        Args:
            labels_df: DataFrame with weak labels
            output_path: Path to save labels
        """
        labels_df.to_csv(output_path, index=False)
        print(f"Weak labels saved to: {output_path}")


def create_weak_labeler(config: Dict[str, any]) -> WeakLabeler:
    """Factory function to create weak labeler from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured WeakLabeler instance
    """
    weak_labels_cfg = config.get("ranker", {}).get("weak_labels", {})

    # Check environment variable for approval mode
    require_approval = os.getenv("WEAK_LABEL_APPROVAL", "auto").lower() == "manual"

    return WeakLabeler(
        priority_weight=weak_labels_cfg.get("priority_weight", 0.4),
        status_weight=weak_labels_cfg.get("status_weight", 0.3),
        hygiene_weight=weak_labels_cfg.get("hygiene_weight", 0.3),
        require_human_approval=require_approval,
    )
