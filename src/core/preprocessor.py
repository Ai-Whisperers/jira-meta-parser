"""Data augmentation and preprocessing for training data enrichment.

This module creates synthetic variations of JIRA issues to improve model
generalization and robustness, especially when training data is limited.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DataPreprocessor:
    """Creates synthetic variations of training data for augmentation."""

    def __init__(
        self,
        augmentation_factor: int = 2,
        text_perturbation: bool = True,
        priority_shuffle: bool = True,
        status_variation: bool = True,
        output_dir: str = "./variations",
        require_human_approval: bool = False,
    ):
        """Initialize data preprocessor.

        Args:
            augmentation_factor: How many variations to create per original issue (1-5)
            text_perturbation: Enable text-based augmentation (synonym, paraphrase)
            priority_shuffle: Create variations with different priorities
            status_variation: Create variations with different statuses
            output_dir: Directory to save variation artifacts
            require_human_approval: If True, prompts for human review before generating
        """
        self.augmentation_factor = min(max(1, augmentation_factor), 5)  # Limit 1-5
        self.text_perturbation = text_perturbation
        self.priority_shuffle = priority_shuffle
        self.status_variation = status_variation
        self.output_dir = Path(output_dir)
        self.require_human_approval = require_human_approval

        # Augmentation mappings
        self.priority_variations = {
            "critical": ["highest", "high"],
            "highest": ["critical", "high"],
            "high": ["highest", "medium"],
            "medium": ["high", "low"],
            "low": ["medium", "lowest"],
            "lowest": ["low", "trivial"],
        }

        self.status_variations = {
            "backlog": ["to do", "new"],
            "to do": ["backlog", "ready"],
            "ready": ["to do", "selected for development"],
            "in progress": ["in development"],
            "blocked": ["on hold"],
        }

    def augment_dataset(
        self,
        features_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Create augmented dataset with synthetic variations.

        Args:
            features_df: Original features DataFrame
            labels_df: Optional labels DataFrame (will be replicated for variations)

        Returns:
            Tuple of (augmented_features_df, augmented_labels_df)
        """
        # Check environment variable for augmentation control
        aug_factor_env = os.getenv("AUGMENTATION_FACTOR", str(self.augmentation_factor))
        try:
            self.augmentation_factor = int(aug_factor_env)
        except ValueError:
            pass

        # Human approval check
        if self.require_human_approval or os.getenv("HUMAN_REVIEW_MODE", "false").lower() == "true":
            if not self._request_approval(features_df):
                print("Augmentation cancelled by user")
                return features_df, labels_df

        print(f"\nGenerating {self.augmentation_factor}x augmentation...")

        augmented_features = [features_df.copy()]
        augmented_labels = [labels_df.copy()] if labels_df is not None else []

        for variation_idx in range(1, self.augmentation_factor):
            print(f"  Creating variation {variation_idx}...")

            variation_features = features_df.copy()

            # Apply augmentations
            if self.text_perturbation:
                variation_features = self._augment_text(variation_features, variation_idx)

            if self.priority_shuffle:
                variation_features = self._augment_priority(variation_features, variation_idx)

            if self.status_variation:
                variation_features = self._augment_status(variation_features, variation_idx)

            # Update keys to make them unique
            variation_features["key"] = variation_features["key"] + f"_aug{variation_idx}"

            augmented_features.append(variation_features)

            # Replicate labels if provided
            if labels_df is not None:
                variation_labels = labels_df.copy()
                variation_labels["key"] = variation_labels["key"] + f"_aug{variation_idx}"
                augmented_labels.append(variation_labels)

        # Combine all variations
        final_features = pd.concat(augmented_features, ignore_index=True)
        final_labels = pd.concat(augmented_labels, ignore_index=True) if augmented_labels else None

        print(f"\nAugmentation complete:")
        print(f"  Original: {len(features_df)} issues")
        print(f"  Augmented: {len(final_features)} issues ({self.augmentation_factor}x)")

        # Save to variations directory
        self._save_variations(final_features, final_labels)

        return final_features, final_labels

    def _augment_text(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Apply text-based augmentation (simple perturbations).

        Args:
            df: Features DataFrame
            seed: Random seed for reproducibility

        Returns:
            Augmented DataFrame
        """
        random.seed(seed)

        # Simple synonym replacement for common terms
        synonyms = {
            "implement": ["develop", "build", "create"],
            "fix": ["resolve", "repair", "correct"],
            "update": ["modify", "change", "revise"],
            "add": ["include", "insert", "incorporate"],
            "remove": ["delete", "eliminate", "drop"],
            "bug": ["issue", "defect", "problem"],
            "feature": ["functionality", "capability", "enhancement"],
        }

        for idx, row in df.iterrows():
            summary = row.get("summary_txt", "")

            # Random synonym replacement (10% chance per word)
            words = summary.lower().split()
            for i, word in enumerate(words):
                if word in synonyms and random.random() < 0.1:
                    words[i] = random.choice(synonyms[word])

            df.at[idx, "summary_txt"] = " ".join(words)

        return df

    def _augment_priority(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Augment priority field with variations.

        Args:
            df: Features DataFrame
            seed: Random seed

        Returns:
            Augmented DataFrame
        """
        random.seed(seed)

        for idx, row in df.iterrows():
            priority = row.get("priority", "medium").lower()

            # 30% chance to vary priority
            if priority in self.priority_variations and random.random() < 0.3:
                df.at[idx, "priority"] = random.choice(self.priority_variations[priority])

        return df

    def _augment_status(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Augment status field with variations.

        Args:
            df: Features DataFrame
            seed: Random seed

        Returns:
            Augmented DataFrame
        """
        random.seed(seed)

        for idx, row in df.iterrows():
            status = row.get("status", "backlog").lower()

            # 20% chance to vary status
            if status in self.status_variations and random.random() < 0.2:
                df.at[idx, "status"] = random.choice(self.status_variations[status])

        return df

    def _request_approval(self, df: pd.DataFrame) -> bool:
        """Request human approval for augmentation.

        Args:
            df: Original DataFrame

        Returns:
            True if approved, False otherwise
        """
        print("\n" + "=" * 80)
        print("DATA AUGMENTATION - HUMAN REVIEW REQUIRED")
        print("=" * 80)
        print(f"\nOriginal dataset: {len(df)} issues")
        print(f"Augmentation factor: {self.augmentation_factor}x")
        print(f"Result: {len(df) * self.augmentation_factor} total issues")
        print(f"\nAugmentation techniques:")
        print(f"  - Text perturbation: {'✓' if self.text_perturbation else '✗'}")
        print(f"  - Priority shuffle: {'✓' if self.priority_shuffle else '✗'}")
        print(f"  - Status variation: {'✓' if self.status_variation else '✗'}")

        response = input("\nProceed with augmentation? (yes/no): ").lower()
        return response == "yes"

    def _save_variations(
        self,
        features_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Save augmented variations to output directory.

        Args:
            features_df: Augmented features
            labels_df: Augmented labels (optional)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        features_path = self.output_dir / "augmented_features.parquet"
        features_df.to_parquet(features_path, index=False)
        print(f"\nAugmented features saved to: {features_path}")

        # Save labels if provided
        if labels_df is not None:
            labels_path = self.output_dir / "augmented_labels.csv"
            labels_df.to_csv(labels_path, index=False)
            print(f"Augmented labels saved to: {labels_path}")

        # Save metadata
        metadata = {
            "original_count": len(features_df) // self.augmentation_factor,
            "augmented_count": len(features_df),
            "augmentation_factor": self.augmentation_factor,
            "techniques": {
                "text_perturbation": self.text_perturbation,
                "priority_shuffle": self.priority_shuffle,
                "status_variation": self.status_variation,
            },
        }

        import json
        metadata_path = self.output_dir / "augmentation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {metadata_path}")


def create_preprocessor(config: Dict[str, any]) -> DataPreprocessor:
    """Factory function to create preprocessor from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured DataPreprocessor instance
    """
    preprocessing_cfg = config.get("preprocessing", {})

    # Check environment variable for human review
    require_approval = os.getenv("HUMAN_REVIEW_MODE", "false").lower() == "true"

    return DataPreprocessor(
        augmentation_factor=preprocessing_cfg.get("augmentation_factor", 2),
        text_perturbation=preprocessing_cfg.get("text_perturbation", True),
        priority_shuffle=preprocessing_cfg.get("priority_shuffle", True),
        status_variation=preprocessing_cfg.get("status_variation", True),
        output_dir=preprocessing_cfg.get("output_dir", "./variations"),
        require_human_approval=require_approval,
    )
