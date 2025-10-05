# JIRA Ticket Meta Parser

**Production-ready ML pipeline for validating and ranking JIRA backlogs**

Transforms chaotic JIRA exports into clean, deterministically ordered backlogs using:
- **Dual-format validation** (XML + CSV)
- **Semantic embeddings** (all-MiniLM-L6-v2)
- **Fast retrieval** (FAISS IVF-PQ)
- **Learning-to-rank** (LightGBM LambdaMART)
- **Optional re-ranking** (ColBERT-v2)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use editable install
pip install -e .
```

### Usage (Production)

```bash
# Simple one-command execution
python -m src.cli.prod --input datasets/JIRA.xml --output backlog.csv --verbose

# Or using the installed command
jira-validate --input JIRA.xml --output clean_backlog.csv
```

### Usage (Development)

```bash
# Run individual stages
python -m src.cli.dev validate datasets/JIRA.xml
python -m src.cli.dev extract datasets/JIRA.xml
python -m src.cli.dev embed
python -m src.cli.dev index
python -m src.cli.dev full datasets/JIRA.xml

# Check pipeline status
python -m src.cli.dev status

# Clean artifacts
python -m src.cli.dev clean
```

---

## Architecture

### Pipeline Flow

```
JIRA.xml/CSV
    ↓
[1] Validator (O(n), streaming)
    ↓
backbone_report.csv + backbone_summary.json
    ↓
[2] Feature Extractor (variability only)
    ↓
variability_features.parquet
    ↓
[3] Embedder (all-MiniLM-L6-v2, 384-D)
    ↓
embeddings.parquet
    ↓
[4] FAISS Indexer (IVF-PQ, nlist=4096, m=16)
    ↓
faiss_index.ivf + .keys.npy
    ↓
[5] Ranker (LightGBM LambdaMART)
    ↓
ranked_backlog.csv
    ↓
[6] Re-ranker (ColBERT-v2, optional, top-50)
    ↓
clean_backlog.csv (final output)
```

### Directory Structure

```
jira-ticket-meta-parser/
├── config/
│   └── default.yaml          # All configuration (zero hardcoding)
├── src/
│   ├── core/                  # Pipeline stages
│   │   ├── validator.py       # Backbone validation
│   │   ├── features.py        # Feature extraction
│   │   ├── embedder.py        # Embedding generation
│   │   ├── indexer.py         # FAISS indexing
│   │   ├── ranker.py          # LambdaMART ranking
│   │   └── reranker.py        # ColBERT re-ranking
│   ├── adapters/              # Format readers
│   │   ├── xml_adapter.py     # Streaming XML parser
│   │   └── csv_adapter.py     # Chunked CSV parser
│   ├── utils/                 # Utilities
│   │   ├── config.py          # Configuration management
│   │   ├── logger.py          # Logging + benchmarking
│   │   ├── artifacts.py       # Output management
│   │   └── text.py            # Text cleaning
│   ├── cli/                   # CLI interfaces
│   │   ├── dev.py             # Developer CLI
│   │   └── prod.py            # Production CLI
│   └── pipeline.py            # Main orchestrator
├── artifacts/                 # All outputs (versioned)
├── benchmarks/                # Performance metrics
├── logs/                      # Execution logs
├── models/                    # Pre-downloaded models
├── datasets/               # Input data
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Configuration

All parameters are managed via `config/default.yaml`. **No hardcoding.**

### ⚠️ Important: Data-Driven Configuration

**The default configuration is optimized for the included 602-issue dataset.** If using your own JIRA data, analyze it first:

```bash
# Generate data analysis report
python -c "from local-reports import DATA_ANALYSIS_REPORT"  # See report for your data
```

**Key optimization learnings from dataset analysis:**

1. **Adjust weak labeling weights** based on your data variance:
   - If priority is uniform (90% same value) → reduce `priority_weight`
   - If status has good coverage → increase `status_weight`
   - Hygiene usually has best differentiation → increase `hygiene_weight`

2. **Map ALL status values** from your JIRA export to avoid unmapped defaults

3. **Update CSV column mapping** for custom fields (Story Points, Epic Name, etc.)

**Current optimized config (based on 602-issue analysis):**
```yaml
ranker:
  weak_labels:
    priority_weight: 0.2  # Reduced - 90% Medium priority
    status_weight: 0.4    # Increased - full coverage
    hygiene_weight: 0.4   # Increased - high variance
    status_scores:
      "in progress": 4.0
      "deploy": 4.0
      "user testing": 3.5
      "approved to start": 3.0
      "in evaluation": 2.5
      "backlog": 2.0
      "rejected": 1.0
      "blocked": 1.0
      "done": 0.0
```

### Key Sections

**Validator:**
- Key regex, date formats, required fields
- Link integrity policy
- CSV column mapping (update for your export format!)

**Features:**
- Text cleaning, categorical fields
- Max text lengths

**Embeddings:**
- Model path, dimension, device (cpu/cuda)

**FAISS:**
- Index type (IVF-PQ), nlist, m, nprobe
- GPU usage

**Ranker (LambdaMART):**
- Objective, metric, hyperparameters
- Grouping strategy (epic/sprint)
- **Weak labels:** Priority/status/hygiene weights + scoring maps

**Preprocessing (Data Augmentation):**
- Augmentation factor (1-5x dataset expansion)
- Text perturbation, priority shuffle, status variation
- Output directory for synthetic variations

**Re-ranker (ColBERT):**
- Enabled/disabled
- Top-K, blend weight

**Artifacts:**
- Output paths, versioning

**Logging:**
- Level, format, benchmark tracking

---

## Key Features

### 1. Dual-Format Support
- **XML**: Streaming parser (handles RSS and `<issues>` format)
- **CSV**: Chunked reader (configurable column mapping)
- **Identical output** regardless of input format

### 2. O(n) Validation
- Single-pass streaming validation
- Enforces backbone schema (keys, dates, types)
- Fail-fast on critical errors
- Link and parent integrity checks

### 3. Separation of Concerns
- **Backbone** (immutable schema) vs. **Variability** (ML features)
- Clean artifacts at each stage
- Reproducible with hash-based versioning

### 4. Production-Grade Logging
- Structured JSON logs
- Automatic benchmarking (duration, memory, metrics)
- Per-stage performance tracking

### 5. Two CLI Interfaces
- **Production** (`src.cli.prod`): Single command, minimal output
- **Developer** (`src.cli.dev`): Stage-by-stage control, debugging

---

## Performance Targets

(From `metacontext.md`)

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 10-50k tickets in <5 min | 1× RTX-3050 + CPU |
| FAISS Recall@K | ≥ 0.95 | vs. brute-force on 2k sample |
| NDCG@10 | ≥ 0.85 | On weak labels |
| PM Approval@20 | ≥ 80% | Human-in-loop validation |
| Kendall τ | ≥ 0.60 | Stability between runs |

---

## Artifacts

All outputs are saved to `artifacts/` with optional hash-based versioning:

| Category | Files | Format |
|----------|-------|--------|
| validation | `backbone_report.csv`, `backbone_summary.json` | CSV, JSON |
| features | `variability_features.parquet` | Parquet |
| embeddings | `embeddings.parquet` | Parquet |
| indices | `faiss_index.ivf`, `faiss_index.keys.npy` | FAISS, NumPy |
| models | `ltr_model.txt`, `feature_map.json` | LightGBM, JSON |
| backlogs | `clean_backlog.csv` | CSV |

Each artifact includes `.meta.json` with creation timestamp, row/column counts, etc.

---

## Development Workflow

### 1. Initial Setup

```bash
# Clone and install
git clone <repo>
cd jira-ticket-meta-parser
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/default.yaml` for your environment:
- Model paths (CPU vs. GPU)
- FAISS parameters (adjust `nlist` for dataset size)
- Ranker hyperparameters

### 3. Run Pipeline

```bash
# Full pipeline (development mode)
python -m src.cli.dev full datasets/JIRA.xml --skip-training

# Production mode (one command)
python -m src.cli.prod -i datasets/JIRA.xml -o output.csv -v
```

### 4. Check Results

```bash
# View artifacts
python -m src.cli.dev status

# Inspect benchmarks
cat benchmarks/benchmarks_*.json
```

### 5. Training (When Labels Available)

```python
from src.pipeline import JIRAPipeline
from src.utils import Config

config = Config()
pipeline = JIRAPipeline(config.to_dict())

# Load data and labels
features_df = pipeline._load_cached("features", "variability_features")
embeddings_df = pipeline._load_cached("embeddings", "embeddings")
validation_df = pipeline._load_cached("validation", "backbone_report")

# Prepare features
X, groups = pipeline.ranker.prepare_features(
    features_df, embeddings_df, validation_df
)

# Train with your labels
y = your_labels  # Relevance scores
metrics = pipeline.ranker.train(X, y, groups)

# Save model
pipeline.ranker.save("artifacts/models/ltr_model.txt")
```

---

## Extending the Pipeline

### Add Custom Validation Rules

Edit `config/default.yaml`:

```yaml
validator:
  required_fields:
    - key
    - summary
    - your_custom_field
```

### Add Custom Features

Edit `src/core/features.py`:

```python
def _extract_row(self, row):
    # ... existing code ...
    features["your_custom_feature"] = compute_custom(row)
    return features
```

### Add Guardrails

Edit `config/default.yaml`:

```yaml
guardrails:
  enabled: true
  rules:
    - name: "your_custom_rule"
      condition: "field == 'value'"
      action: "demote_by_10"
```

---

## Troubleshooting

### Issue: Validation fails with "missing required fields"

**Solution:** Check `artifacts/validation/backbone_report.csv` for details. Update `config/default.yaml` if needed.

### Issue: Embeddings stage is slow

**Solution:**
- Reduce `batch_size` in config (lower memory, slower)
- Use GPU: change `device: "cuda"` and install `pip install faiss-gpu`

### Issue: FAISS index build fails

**Solution:**
- For small datasets (<1000 issues), reduce `nlist` in config
- Check that embeddings have correct dimension (384 for MiniLM)

### Issue: Ranking produces unexpected order

**Solution:**
- Pipeline uses **weak labels** by default (heuristic scoring)
- For production, train LambdaMART with **real labels** (see Training section)

---

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## License

[Your License Here]

---

## References

Based on:
- **metacontext.md**: Full pipeline specification
- **validator-context.md**: Dual-format validator architecture

Models:
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ColBERT-v2](https://github.com/stanford-futuredata/ColBERT) (optional)
