"""Basic sanity tests to verify installation and imports."""

import pytest


def test_imports():
    """Test that all core modules import successfully."""
    from src.core import (
        BackboneValidator,
        ColBERTReranker,
        Embedder,
        FAISSIndexer,
        FeatureExtractor,
        LambdaMARTRanker,
    )
    from src.utils import ArtifactManager, Config, StructuredLogger

    assert BackboneValidator is not None
    assert FeatureExtractor is not None
    assert Embedder is not None
    assert FAISSIndexer is not None
    assert LambdaMARTRanker is not None
    assert ColBERTReranker is not None
    assert Config is not None
    assert ArtifactManager is not None
    assert StructuredLogger is not None


def test_config_load():
    """Test that default config loads successfully."""
    from src.utils import Config

    config = Config()
    assert config is not None
    assert "validator" in config.to_dict()
    assert "embeddings" in config.to_dict()
    assert "faiss" in config.to_dict()
    assert "ranker" in config.to_dict()


def test_pipeline_init():
    """Test that pipeline initializes successfully."""
    from src.pipeline import JIRAPipeline
    from src.utils import Config

    config = Config()
    pipeline = JIRAPipeline(config.to_dict())

    assert pipeline is not None
    assert pipeline.validator is not None
    assert pipeline.feature_extractor is not None
    assert pipeline.embedder is not None
    assert pipeline.indexer is not None
    assert pipeline.ranker is not None


def test_xml_adapter():
    """Test XML adapter basic functionality."""
    from src.adapters import XMLAdapter

    adapter = XMLAdapter()
    assert adapter is not None
    assert hasattr(adapter, "parse")


def test_csv_adapter():
    """Test CSV adapter basic functionality."""
    from src.adapters import CSVAdapter

    adapter = CSVAdapter()
    assert adapter is not None
    assert hasattr(adapter, "parse")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
