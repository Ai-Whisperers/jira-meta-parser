"""JIRA Ticket Meta Parser - Production-ready ML pipeline."""

__version__ = "1.0.0"

from .pipeline import JIRAPipeline
from .utils import Config

__all__ = ["JIRAPipeline", "Config"]
