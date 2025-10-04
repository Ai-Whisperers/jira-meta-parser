"""Structured logging and benchmarking utilities."""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class StructuredLogger:
    """Production-grade logger with benchmark tracking."""

    def __init__(
        self,
        name: str,
        log_dir: str = "./logs",
        log_file: str = "pipeline.log",
        level: str = "INFO",
        structured: bool = True,
        track_performance: bool = True,
        benchmark_dir: str = "./benchmarks",
    ):
        """Initialize logger.

        Args:
            name: Logger name (typically module name)
            log_dir: Directory for log files
            log_file: Log filename
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            structured: Use structured JSON logging
            track_performance: Enable benchmark tracking
            benchmark_dir: Directory for benchmark outputs
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.structured = structured
        self.track_performance = track_performance
        self.benchmark_dir = Path(benchmark_dir)

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.track_performance:
            self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # File handler
        fh = logging.FileHandler(self.log_dir / self.log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper()))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))

        # Formatter
        if structured:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": %(message)s}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.benchmarks: Dict[str, list] = {}

    def _format_msg(self, msg: str, **kwargs) -> str:
        """Format message for structured logging."""
        if self.structured and kwargs:
            data = {"message": msg, **kwargs}
            return json.dumps(data)
        return msg

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_msg(msg, **kwargs))

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_msg(msg, **kwargs))

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_msg(msg, **kwargs))

    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_msg(msg, **kwargs))

    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_msg(msg, **kwargs))

    def benchmark(self, stage: str, duration: float, **metrics):
        """Record benchmark data.

        Args:
            stage: Pipeline stage name
            duration: Execution duration in seconds
            **metrics: Additional metrics to track
        """
        if not self.track_performance:
            return

        if stage not in self.benchmarks:
            self.benchmarks[stage] = []

        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "duration_sec": round(duration, 4),
            **metrics,
        }

        self.benchmarks[stage].append(record)

        # Also log it
        self.info(
            f"Benchmark: {stage}",
            duration_sec=round(duration, 4),
            **metrics,
        )

    def save_benchmarks(self, suffix: str = ""):
        """Save all benchmarks to JSON file.

        Args:
            suffix: Optional suffix for filename
        """
        if not self.track_performance or not self.benchmarks:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmarks_{timestamp}{suffix}.json"
        filepath = self.benchmark_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.benchmarks, f, indent=2)

        self.info(f"Benchmarks saved", filepath=str(filepath))


def benchmark_stage(logger: Optional[StructuredLogger] = None, stage_name: str = None):
    """Decorator to automatically benchmark a function.

    Args:
        logger: StructuredLogger instance
        stage_name: Override stage name (default: function name)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = stage_name or func.__name__

            if logger:
                logger.info(f"Starting stage: {name}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                if logger:
                    logger.benchmark(name, duration, status="success")
                    logger.info(f"Completed stage: {name}", duration_sec=round(duration, 4))

                return result

            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.benchmark(name, duration, status="failed", error=str(e))
                    logger.error(f"Failed stage: {name}", error=str(e), duration_sec=round(duration, 4))
                raise

        return wrapper

    return decorator


def create_logger(config: Dict[str, Any], name: str = __name__) -> StructuredLogger:
    """Factory function to create logger from config.

    Args:
        config: Configuration dictionary
        name: Logger name

    Returns:
        Configured StructuredLogger instance
    """
    logging_cfg = config.get("logging", {})

    return StructuredLogger(
        name=name,
        log_dir=logging_cfg.get("log_dir", "./logs"),
        log_file=logging_cfg.get("log_file", "pipeline.log"),
        level=logging_cfg.get("level", "INFO"),
        structured=logging_cfg.get("format", "structured") == "structured",
        track_performance=logging_cfg.get("track_performance", True),
        benchmark_dir=logging_cfg.get("benchmark_dir", "./benchmarks"),
    )
