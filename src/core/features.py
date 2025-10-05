"""Variability feature extraction - separates content from schema."""

from typing import Dict, List, Optional

import pandas as pd

from ..utils.text import extract_text_features, normalize_categorical


class FeatureExtractor:
    """Extracts variability features from validated issues."""

    def __init__(
        self,
        categorical_fields: List[str],
        strip_html: bool = True,
        max_summary_len: Optional[int] = None,
        max_description_len: Optional[int] = None,
    ):
        """Initialize feature extractor.

        Args:
            categorical_fields: Fields to treat as categorical (type, status, priority)
            strip_html: Whether to strip HTML from text fields
            max_summary_len: Max length for summary text
            max_description_len: Max length for description text
        """
        self.categorical_fields = categorical_fields
        self.strip_html = strip_html
        self.max_summary_len = max_summary_len
        self.max_description_len = max_description_len

    def extract(self, rows: List[Dict[str, any]]) -> pd.DataFrame:
        """Extract variability features from issue rows.

        Args:
            rows: List of canonical issue dictionaries

        Returns:
            DataFrame with variability features
        """
        features = []

        for row in rows:
            feature_dict = self._extract_row(row)
            features.append(feature_dict)

        return pd.DataFrame(features)

    def _extract_row(self, row: Dict[str, any]) -> Dict[str, any]:
        """Extract features from a single row.

        Args:
            row: Canonical issue dictionary

        Returns:
            Feature dictionary
        """
        # Text features (cleaned)
        text_features = extract_text_features(
            summary=row.get("summary"),
            description=row.get("description"),
        )

        # Truncate if needed
        if self.max_summary_len:
            text_features["summary_txt"] = text_features["summary_txt"][:self.max_summary_len]
        if self.max_description_len:
            text_features["description_txt"] = text_features["description_txt"][:self.max_description_len]

        # Categorical anchors (normalized)
        categorical = {}
        for field in self.categorical_fields:
            categorical[field] = normalize_categorical(row.get(field))

        # Count features
        counts = {
            "label_count": len(row.get("labels", [])),
            "component_count": len(row.get("components", [])),
            "customfield_count": row.get("customfield_count", 0),
            "link_count": len(row.get("issuelinks", [])),
        }

        # Dirty flags (missing required business data)
        dirty_flags = self._extract_dirty_flags(row)

        # Combine all features
        features = {
            "key": row.get("key"),
            **text_features,
            **categorical,
            **counts,
            **dirty_flags,
        }

        # Add optional fields for grouping
        features["epic"] = row.get("epic_link") or row.get("parent")
        features["sprint"] = row.get("sprint")

        return features

    def _extract_dirty_flags(self, row: Dict[str, any]) -> Dict[str, int]:
        """Extract 'dirty' business logic flags.

        Args:
            row: Canonical issue dictionary

        Returns:
            Dictionary of binary flags
        """
        flags = {}

        # Missing acceptance criteria (common JIRA custom field pattern)
        # Heuristic: if description is very short or empty
        desc = row.get("description") or ""
        flags["flag_missing_ac"] = 1 if len(desc.strip()) < 20 else 0

        # Missing assignee (check for "Unassigned" or empty)
        assignee = row.get("assignee") or ""
        flags["assignee_empty"] = 1 if not assignee or "unassigned" in assignee.lower() else 0

        # Missing story points - IMPROVED to check actual Story Points field
        # Supports both direct story_points field and customfield IDs
        story_points = row.get("story_points") or row.get("customfield_10016") or ""
        flags["storypoints_empty"] = 1 if not str(story_points).strip() or str(story_points) == "0" else 0

        return flags


def create_feature_extractor(config: Dict[str, any]) -> FeatureExtractor:
    """Factory function to create feature extractor from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured FeatureExtractor instance
    """
    features_cfg = config.get("features", {})

    return FeatureExtractor(
        categorical_fields=features_cfg.get("categorical_fields", ["type", "status", "priority"]),
        strip_html=features_cfg.get("strip_html", True),
        max_summary_len=features_cfg.get("max_summary_len"),
        max_description_len=features_cfg.get("max_description_len"),
    )
