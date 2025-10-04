"""Configuration management - zero hardcoding, all from YAML."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration loader with validation and path resolution."""

    def __init__(self, config_path: str = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses default.yaml
        """
        if config_path is None:
            # Default to config/default.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "default.yaml"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config: Dict[str, Any] = yaml.safe_load(f)

        # Resolve paths relative to project root
        self._resolve_paths()

    def _resolve_paths(self):
        """Convert relative paths to absolute paths."""
        project_root = Path(__file__).parent.parent.parent

        # Resolve embedding model path
        if "embeddings" in self._config and "model_path" in self._config["embeddings"]:
            model_path = self._config["embeddings"]["model_path"]
            if not Path(model_path).is_absolute():
                self._config["embeddings"]["model_path"] = str(
                    (project_root / model_path).resolve()
                )

        # Resolve reranker model path
        if "reranker" in self._config and "model_path" in self._config["reranker"]:
            model_path = self._config["reranker"]["model_path"]
            if not Path(model_path).is_absolute():
                self._config["reranker"]["model_path"] = str(
                    (project_root / model_path).resolve()
                )

        # Resolve artifacts base directory
        if "artifacts" in self._config and "base_dir" in self._config["artifacts"]:
            base_dir = self._config["artifacts"]["base_dir"]
            if not Path(base_dir).is_absolute():
                self._config["artifacts"]["base_dir"] = str((project_root / base_dir).resolve())

        # Resolve logging directory
        if "logging" in self._config and "log_dir" in self._config["logging"]:
            log_dir = self._config["logging"]["log_dir"]
            if not Path(log_dir).is_absolute():
                self._config["logging"]["log_dir"] = str((project_root / log_dir).resolve())

        # Resolve benchmark directory
        if "logging" in self._config and "benchmark_dir" in self._config["logging"]:
            bench_dir = self._config["logging"]["benchmark_dir"]
            if not Path(bench_dir).is_absolute():
                self._config["logging"]["benchmark_dir"] = str(
                    (project_root / bench_dir).resolve()
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Dot-separated path (e.g., 'validator.key_regex')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to top-level keys."""
        return self._config[key]

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()
