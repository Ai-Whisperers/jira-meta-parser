"""Artifact management - deterministic outputs with versioning."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ArtifactManager:
    """Manages pipeline artifacts with versioning and metadata."""

    def __init__(self, base_dir: str, version_artifacts: bool = True):
        """Initialize artifact manager.

        Args:
            base_dir: Base directory for all artifacts
            version_artifacts: Whether to add hash-based versioning
        """
        self.base_dir = Path(base_dir)
        self.version_artifacts = version_artifacts

        # Create subdirectories
        self.dirs = {
            "validation": self.base_dir / "validation",
            "features": self.base_dir / "features",
            "embeddings": self.base_dir / "embeddings",
            "indices": self.base_dir / "indices",
            "models": self.base_dir / "models",
            "backlogs": self.base_dir / "backlogs",
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        category: str,
        name: str,
        format: str = "parquet",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save DataFrame with optional versioning.

        Args:
            df: DataFrame to save
            category: Category (validation, features, etc.)
            name: Base filename (without extension)
            format: File format (parquet, csv)
            metadata: Optional metadata to save alongside

        Returns:
            Path to saved file
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")

        # Generate filename
        if self.version_artifacts:
            content_hash = self._hash_dataframe(df)
            filename = f"{name}_{content_hash[:8]}.{format}"
        else:
            filename = f"{name}.{format}"

        filepath = self.dirs[category] / filename

        # Save data
        if format == "parquet":
            df.to_parquet(filepath, index=False, engine="pyarrow")
        elif format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save metadata if provided
        if metadata:
            meta_path = filepath.with_suffix(".meta.json")
            self._save_metadata(meta_path, df, metadata)

        return filepath

    def save_json(
        self,
        data: Dict[str, Any],
        category: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save JSON data.

        Args:
            data: Dictionary to save
            category: Category (validation, features, etc.)
            name: Base filename (without extension)
            metadata: Optional metadata

        Returns:
            Path to saved file
        """
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")

        # Generate filename
        if self.version_artifacts:
            content_hash = self._hash_dict(data)
            filename = f"{name}_{content_hash[:8]}.json"
        else:
            filename = f"{name}.json"

        filepath = self.dirs[category] / filename

        # Save data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Save metadata if provided
        if metadata:
            meta_path = filepath.with_suffix(".meta.json")
            self._save_metadata(meta_path, data, metadata)

        return filepath

    def _save_metadata(self, meta_path: Path, data: Any, metadata: Dict[str, Any]):
        """Save metadata file.

        Args:
            meta_path: Path to metadata file
            data: Original data (for stats)
            metadata: User-provided metadata
        """
        meta = {
            "created_at": datetime.now().isoformat(),
            "type": type(data).__name__,
            **metadata,
        }

        # Add size info
        if isinstance(data, pd.DataFrame):
            meta["rows"] = len(data)
            meta["columns"] = list(data.columns)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Generate hash of DataFrame content.

        Args:
            df: DataFrame to hash

        Returns:
            SHA256 hash hex string
        """
        # Hash column names and first few rows for efficiency
        content = f"{list(df.columns)}{df.head(100).to_json()}"
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Generate hash of dictionary content.

        Args:
            data: Dictionary to hash

        Returns:
            SHA256 hash hex string
        """
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_latest(self, category: str, name: str, format: str = None) -> Optional[Path]:
        """Get latest version of an artifact.

        Args:
            category: Category to search in
            name: Base filename to search for
            format: Optional format filter

        Returns:
            Path to latest file, or None if not found
        """
        if category not in self.dirs:
            return None

        pattern = f"{name}_*.{format}" if format else f"{name}_*"
        files = sorted(self.dirs[category].glob(pattern), key=lambda p: p.stat().st_mtime)

        return files[-1] if files else None
