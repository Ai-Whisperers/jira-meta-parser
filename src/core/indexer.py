"""FAISS indexing for fast similarity search (IVF-PQ)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd


class FAISSIndexer:
    """FAISS IVF-PQ indexer for embedding search."""

    def __init__(
        self,
        dimension: int = 384,
        nlist: int = 4096,
        m: int = 16,
        nbits: int = 8,
        nprobe: int = 16,
        use_gpu: bool = False,
    ):
        """Initialize FAISS indexer.

        Args:
            dimension: Embedding dimension
            nlist: Number of coarse quantizer clusters
            m: Number of PQ sub-vectors
            nbits: Bits per PQ sub-vector
            nprobe: Number of clusters to scan at query time
            use_gpu: Whether to use GPU (requires faiss-gpu)
        """
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self.index: Optional[faiss.Index] = None
        self.key_map: List[str] = []  # Maps index position to issue key

    def build(self, embeddings_df: pd.DataFrame) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings_df: DataFrame with 'key' and 'embedding' columns
        """
        # Extract embeddings and keys
        embeddings = np.vstack(embeddings_df["embedding"].values).astype("float32")
        self.key_map = embeddings_df["key"].tolist()

        n_vectors = len(embeddings)

        # Adjust nlist if dataset is small
        nlist = min(self.nlist, n_vectors // 10) if n_vectors < 10000 else self.nlist

        # Create quantizer
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create IVF-PQ index
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            nlist,
            self.m,
            self.nbits,
        )

        # Train index
        self.index.train(embeddings)

        # Add vectors
        self.index.add(embeddings)

        # Set search parameters
        self.index.nprobe = self.nprobe

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Warning: GPU transfer failed: {e}. Using CPU.")

    def search(
        self, query_embeddings: np.ndarray, k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """Search for top-K nearest neighbors.

        Args:
            query_embeddings: Query vectors (n_queries × dimension)
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (distances, indices, keys)
                - distances: (n_queries × k) distance matrix
                - indices: (n_queries × k) index matrix
                - keys: List of lists of issue keys
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Ensure float32
        query_embeddings = query_embeddings.astype("float32")

        # Search
        distances, indices = self.index.search(query_embeddings, k)

        # Map indices to keys
        keys = [
            [self.key_map[idx] if idx >= 0 else None for idx in row]
            for row in indices
        ]

        return distances, indices, keys

    def save(self, filepath: str) -> None:
        """Save index to disk.

        Args:
            filepath: Path to save index
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Move to CPU if on GPU
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        # Save index
        faiss.write_index(index_to_save, filepath)

        # Save key mapping
        keymap_path = Path(filepath).with_suffix(".keys.npy")
        np.save(keymap_path, np.array(self.key_map, dtype=object))

    def load(self, filepath: str) -> None:
        """Load index from disk.

        Args:
            filepath: Path to index file
        """
        # Load index
        self.index = faiss.read_index(filepath)
        self.index.nprobe = self.nprobe

        # Load key mapping
        keymap_path = Path(filepath).with_suffix(".keys.npy")
        if keymap_path.exists():
            self.key_map = np.load(keymap_path, allow_pickle=True).tolist()

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception as e:
                print(f"Warning: GPU transfer failed: {e}. Using CPU.")

    def get_index_stats(self) -> Dict[str, any]:
        """Get index statistics.

        Returns:
            Dictionary of index stats
        """
        if self.index is None:
            return {"built": False}

        return {
            "built": True,
            "n_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "m": self.m,
            "nprobe": self.nprobe,
            "use_gpu": self.use_gpu,
        }


def create_indexer(config: Dict[str, any]) -> FAISSIndexer:
    """Factory function to create FAISS indexer from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured FAISSIndexer instance
    """
    faiss_cfg = config.get("faiss", {})

    return FAISSIndexer(
        dimension=config.get("embeddings", {}).get("dimension", 384),
        nlist=faiss_cfg.get("nlist", 4096),
        m=faiss_cfg.get("m", 16),
        nbits=faiss_cfg.get("nbits", 8),
        nprobe=faiss_cfg.get("nprobe", 16),
        use_gpu=faiss_cfg.get("use_gpu", False),
    )
