"""Embedding generation using sentence-transformers (all-MiniLM-L6-v2)."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates semantic embeddings for text features."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_path: Optional[str] = None,
        dimension: int = 384,
        normalize_l2: bool = True,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Initialize embedder.

        Args:
            model_name: Hugging Face model name
            model_path: Local path to cached model (if available)
            dimension: Expected embedding dimension
            normalize_l2: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.model_path = model_path
        self.dimension = dimension
        self.normalize_l2 = normalize_l2
        self.batch_size = batch_size
        self.device = device

        # Load model
        if model_path and Path(model_path).exists():
            self.model = SentenceTransformer(model_path, device=device)
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def embed_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for text features.

        Args:
            features_df: DataFrame with 'summary_txt' and 'description_txt' columns

        Returns:
            DataFrame with 'key' and 'embedding' (384-D array) columns
        """
        # Combine summary and description for embedding
        texts = self._prepare_texts(features_df)

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_l2,
        )

        # Verify dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Unexpected embedding dimension: {embeddings.shape[1]} != {self.dimension}"
            )

        # Create DataFrame
        result = pd.DataFrame(
            {
                "key": features_df["key"].values,
                "embedding": list(embeddings),
            }
        )

        return result

    def _prepare_texts(self, features_df: pd.DataFrame) -> List[str]:
        """Prepare text for embedding (summary + description).

        Args:
            features_df: Features DataFrame

        Returns:
            List of combined texts
        """
        texts = []

        for _, row in features_df.iterrows():
            summary = row.get("summary_txt", "")
            description = row.get("description_txt", "")

            # Combine with delimiter
            combined = f"{summary} [SEP] {description}".strip()
            texts.append(combined)

        return texts

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            text: Query text

        Returns:
            Embedding vector (384-D)
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.normalize_l2,
        )[0]

        return embedding


def create_embedder(config: Dict[str, any]) -> Embedder:
    """Factory function to create embedder from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Embedder instance
    """
    embeddings_cfg = config.get("embeddings", {})

    return Embedder(
        model_name=embeddings_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        model_path=embeddings_cfg.get("model_path"),
        dimension=embeddings_cfg.get("dimension", 384),
        normalize_l2=embeddings_cfg.get("normalize_l2", True),
        batch_size=embeddings_cfg.get("batch_size", 32),
        device=embeddings_cfg.get("device", "cpu"),
    )
