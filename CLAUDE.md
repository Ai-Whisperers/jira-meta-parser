# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JIRA Ticket Meta Parser is a production ML pipeline that transforms chaotic JIRA exports into deterministically ranked backlogs. The system uses dual-format validation (XML/CSV), semantic embeddings (all-MiniLM-L6-v2), FAISS indexing, and LightGBM LambdaMART for learning-to-rank.

**Key principle:** Zero hardcoding - all configuration lives in `config/default.yaml`.

## Common Commands

### Development CLI

```bash
# Full pipeline (development mode with stage control)
python -m src.cli.dev full raw-dataset/JIRA.xml

# Individual stages
python -m src.cli.dev validate raw-dataset/JIRA.xml
python -m src.cli.dev extract raw-dataset/JIRA.xml
python -m src.cli.dev embed
python -m src.cli.dev index
python -m src.cli.dev status
python -m src.cli.dev clean

# With environment controls
export WEAK_LABEL_APPROVAL=manual
python -m src.cli.dev full raw-dataset/JIRA.xml
```

### Production CLI

```bash
# Single-command execution (production mode)
python -m src.cli.prod --input raw-dataset/JIRA.xml --output backlog.csv --verbose

# Or using installed command
jira-validate --input JIRA.xml --output clean_backlog.csv
```

### Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_basic.py -v
```

### Code Quality

```bash
# Format code (Black)
black src/ tests/ --line-length 100

# Lint code (Flake8)
flake8 src/ tests/
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Editable install (for development)
pip install -e .

# Optional: GPU support
pip install faiss-gpu  # Replace faiss-cpu
```

## Architecture

### Pipeline Flow

The pipeline follows a strict sequence of stages, each with clean input/output artifacts:

1. **Validation** (`src/core/validator.py`) → `backbone_report.csv` + `backbone_summary.json`
2. **Feature Extraction** (`src/core/features.py`) → `variability_features.parquet`
3. **Weak Labeling** (`src/core/weak_labeler.py`) → `weak_labels.csv` (optional)
4. **Preprocessing/Augmentation** (`src/core/preprocessor.py`) → augmented datasets (optional)
5. **Embeddings** (`src/core/embedder.py`) → `embeddings.parquet` (384-D vectors)
6. **FAISS Indexing** (`src/core/indexer.py`) → `faiss_index.ivf` + `.keys.npy`
7. **Ranking** (`src/core/ranker.py`) → `clean_backlog.csv`
8. **Re-ranking** (`src/core/reranker.py`) → final output (optional, ColBERT)

### Key Design Patterns

**Backbone vs. Variability Separation:**
- **Backbone**: Immutable schema validation (keys, dates, required fields) - handled by `validator.py`
- **Variability**: ML-specific features (text, embeddings, derived metrics) - handled by `features.py`

**Dual-Format Adapters:**
- `src/adapters/xml_adapter.py`: Streaming XML parser (handles RSS and `<issues>` format)
- `src/adapters/csv_adapter.py`: Chunked CSV reader with configurable column mapping
- Both produce identical row dictionaries for downstream processing

**Artifact Management:**
- All outputs go to `artifacts/` with optional hash-based versioning
- Each artifact includes `.meta.json` with creation timestamp, row/column counts
- Managed by `src/utils/artifacts.py` (ArtifactManager)

**Configuration-Driven:**
- Single source of truth: `config/default.yaml`
- Loaded by `src/utils/config.py` (Config class)
- Environment variables override config for human-in-loop controls (see `ENVIRONMENT_VARIABLES.md`)

### Component Interactions

**Pipeline orchestrator** (`src/pipeline.py`):
- `JIRAPipeline` class coordinates all stages
- Each stage decorated with `@benchmark_stage` for automatic performance tracking
- Stages can be skipped via flags (`skip_validation`, `skip_training`)
- Uses `ArtifactManager` for caching and artifact retrieval

**Weak labeling workflow** (`src/core/weak_labeler.py`):
- Generates heuristic relevance scores when real labels unavailable
- Combines priority (0.2), status (0.4), hygiene (0.4) weights
- Environment variable `WEAK_LABEL_APPROVAL=manual` enables human review
- Outputs statistics (mean, std, range) for quality assessment

**Data augmentation** (`src/core/preprocessor.py`):
- Enabled via `config/default.yaml` → `preprocessing.enabled: true`
- Generates synthetic variations (text perturbation, priority shuffle, status variation)
- `AUGMENTATION_FACTOR` env var controls 1-5x dataset expansion
- `HUMAN_REVIEW_MODE=true` adds approval checkpoint

## Configuration Best Practices

### Data-Driven Optimization

**Critical:** The default config is optimized for the included 602-issue dataset. For your own data:

1. **Analyze your dataset first:**
   - Check priority distribution (if 90% same value, reduce `priority_weight`)
   - Check status coverage (map ALL values to avoid defaults)
   - Check hygiene variance (missing fields = good differentiation)

2. **Update CSV column mapping** (`config/default.yaml` → `validator.csv_column_mapping`):
   - Match actual export column names exactly
   - Add custom fields (Story Points, Epic Name, Team, etc.)
   - Wrong mapping = silent failures

3. **Adjust weak labeling weights** based on variance analysis:
   ```yaml
   ranker:
     weak_labels:
       priority_weight: 0.2  # Low if uniform distribution
       status_weight: 0.4    # High if good coverage
       hygiene_weight: 0.4   # High if many incomplete tickets
   ```

### FAISS Index Tuning

- **Small datasets (<1000 issues):** Reduce `nlist` to avoid empty clusters
- **Large datasets (>50k issues):** Increase `nlist` for better recall
- **GPU available:** Set `use_gpu: true` and `pip install faiss-gpu`

### Model Paths

Models are expected in `./models/model-files/`:
- `all-MiniLM-L6-v2/` (sentence embeddings)
- `colbertv2.0/` (optional re-ranker)

Update `config/default.yaml` if models are elsewhere.

## Development Workflow

### Adding Custom Features

Edit `src/core/features.py`:
```python
def _extract_row(self, row):
    # ... existing code ...
    features["your_custom_feature"] = compute_custom(row)
    return features
```

### Adding Custom Validation Rules

Edit `config/default.yaml`:
```yaml
validator:
  required_fields:
    - key
    - summary
    - your_custom_field
```

### Training with Real Labels

When you have ground truth labels:

```python
from src.pipeline import JIRAPipeline
from src.utils.config import Config

config = Config().to_dict()
pipeline = JIRAPipeline(config)

# Load cached artifacts
features_df = pipeline._load_cached("features", "variability_features")
embeddings_df = pipeline._load_cached("embeddings", "embeddings")
validation_df = pipeline._load_cached("validation", "backbone_report")

# Prepare features
X, groups = pipeline.ranker.prepare_features(features_df, embeddings_df, validation_df)

# Train with your labels
y = your_labels  # Relevance scores
metrics = pipeline.ranker.train(X, y, groups)

# Save model
pipeline.ranker.save("artifacts/models/ltr_model.txt")
```

### Debugging Pipeline Stages

```bash
# Run validation only, inspect report
python -m src.cli.dev validate raw-dataset/JIRA.xml
cat artifacts/validation/backbone_report.csv

# Check weak label distribution
python -c "
import pandas as pd
labels = pd.read_csv('artifacts/labels/weak_labels.csv')
print(labels['relevance_score'].describe())
"

# View benchmarks
cat benchmarks/benchmarks_*.json | jq '.validation, .embeddings'
```

## Important Notes

### Configuration Changes

- **Never hardcode** paths, parameters, or thresholds in code
- Add new parameters to `config/default.yaml`
- Use `self.config.get("section", {}).get("key", default)` pattern
- Update `CLAUDE.md` and `README.md` when adding config sections

### Testing Changes

- Write tests in `tests/` for new features
- Use `pytest-benchmark` for performance-critical code
- Validate against both XML and CSV formats
- Check that artifacts are created with correct metadata

### Logging

- Use `self.logger.info()`, `self.logger.error()`, etc.
- Include relevant context (counts, paths, metrics)
- All logs go to `logs/pipeline.log` and stdout
- Benchmarks auto-saved to `benchmarks/` when using `@benchmark_stage`

### Common Pitfalls

1. **CSV column mapping mismatch**: Export format varies by JIRA instance - always check actual column names
2. **Status values unmapped**: Defaults to 1.0 score - causes poor differentiation
3. **Wrong parent field name**: CSV exports use "Parent key" not "parent"
4. **Missing Story Points extraction**: Check actual field name in CSV (can be "Custom field (Story Points)")
5. **FAISS index build fails**: Reduce `nlist` for small datasets or check embedding dimension (must be 384)

## Environment Variables

See `ENVIRONMENT_VARIABLES.md` for human-in-loop controls:

- `WEAK_LABEL_APPROVAL=manual`: Review weak labels before training
- `HUMAN_REVIEW_MODE=true`: Approve augmentation plan
- `AUGMENTATION_FACTOR=2`: Dataset expansion multiplier (1-5)

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 10-50k tickets in <5 min | RTX-3050 + CPU |
| FAISS Recall@K | ≥ 0.95 | vs. brute-force |
| NDCG@10 | ≥ 0.85 | On weak labels |
| PM Approval@20 | ≥ 80% | Human validation |
| Kendall τ | ≥ 0.60 | Run stability |

## Related Documentation

- `README.md`: Full pipeline overview and usage
- `TRAINING_READY.md`: Dataset analysis and training guide
- `QUICKSTART.md`: Step-by-step getting started
- `ARCHITECTURE.md`: Detailed system design
- `ENVIRONMENT_VARIABLES.md`: Human-in-loop controls
- `metacontext.md`: Original specification
- `validator-context.md`: Validation architecture
