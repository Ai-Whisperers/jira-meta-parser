# Architecture Documentation

## Design Principles

### 1. **Zero Hardcoding**
All configuration lives in `config/default.yaml`. Code never contains:
- File paths
- Model names
- Hyperparameters
- Business logic thresholds

### 2. **Separation of Concerns**

**Backbone vs. Variability:**
- **Backbone** = immutable JIRA metadata schema (keys, dates, types)
- **Variability** = content features for ML (text, counts, flags)

This separation ensures models learn **content**, not **meta-structure**.

**Developer vs. User Interfaces:**
- **Developer CLI** (`src.cli.dev`): Full stage control, debugging, experimentation
- **Production CLI** (`src.cli.prod`): Single command, minimal output, production-ready

### 3. **Streaming & O(n) Complexity**

**XML Adapter:**
- Uses `iterparse` to avoid loading entire DOM
- Yields canonical dicts one-by-one
- Memory: O(1) per issue

**CSV Adapter:**
- Chunked reading via pandas
- Configurable chunk size
- Memory: O(chunk_size)

**Validator:**
- Two-pass validation (O(n) each):
  1. Collect all keys
  2. Validate rows + check link integrity
- No O(n²) operations

### 4. **Artifact Versioning**

All outputs include:
- Hash-based filenames (optional, configurable)
- `.meta.json` with creation timestamp, row counts
- Deterministic: same input → same output hash

### 5. **Benchmarking from Day Zero**

Every pipeline stage is decorated with `@benchmark_stage`:
- Automatic duration tracking
- Custom metrics (NDCG, Kendall τ)
- Saved to `benchmarks/` directory

---

## Module Hierarchy

```
src/
├── adapters/        # Input format readers (XML, CSV)
│   ├── xml_adapter.py
│   └── csv_adapter.py
├── core/            # Pipeline stages (independent, testable)
│   ├── validator.py
│   ├── features.py
│   ├── embedder.py
│   ├── indexer.py
│   ├── ranker.py
│   └── reranker.py
├── utils/           # Cross-cutting utilities
│   ├── config.py
│   ├── logger.py
│   ├── artifacts.py
│   └── text.py
├── cli/             # User interfaces
│   ├── dev.py
│   └── prod.py
├── pipeline.py      # Orchestrator (coordinates core modules)
└── __init__.py
```

### Dependency Flow

```
pipeline.py
    ↓
core/* (validator, features, embedder, indexer, ranker, reranker)
    ↓
adapters/* (xml_adapter, csv_adapter)
    ↓
utils/* (config, logger, artifacts, text)
```

**Key invariant:** Core modules never depend on each other directly. Pipeline orchestrates.

---

## Data Flow

### Canonical Issue Dictionary

All adapters (XML, CSV) produce identical output format:

```python
{
    "key": "PROJ-123",
    "summary": "Issue summary",
    "type": "Story",
    "status": "To Do",
    "priority": "High",
    "created": "2025-01-01T10:00:00Z",
    "updated": "2025-01-02T15:30:00Z",
    "description": "Full description...",
    "assignee": "user@example.com",
    "reporter": "reporter@example.com",
    "labels": ["label1", "label2"],
    "components": ["component1"],
    "parent": "PROJ-100",
    "epic_link": "PROJ-50",
    "sprint": "Sprint 1",
    "rank": "1",
    "issuelinks": [
        {"type": "blocks", "direction": "outward", "key": "PROJ-124"}
    ],
    "customfield_count": 5
}
```

### Validation Output

**backbone_report.csv:**

| key | required_ok | key_format_ok | unique_key_ok | created_ok | updated_ok | dates_ok | link_ref_errors | parent_ok | summary | type | status | priority |
|-----|-------------|---------------|---------------|-----------|-----------|---------|----------------|----------|---------|------|--------|----------|
| PROJ-123 | True | True | True | True | True | True | 0 | True | Issue... | Story | To Do | High |

**backbone_summary.json:**

```json
{
  "issues_count": 600,
  "unique_keys_count": 600,
  "errors": {
    "missing_required": 0,
    "bad_key_format": 0,
    "duplicate_key": 0,
    "bad_dates": 0,
    "link_reference_errors": 0,
    "invalid_parent": 0
  },
  "schema_backbone": {
    "required_fields": ["key", "summary", ...],
    "optional_fields": ["description", "assignee", ...]
  }
}
```

### Feature Output

**variability_features.parquet:**

| key | summary_txt | description_txt | summary_len | description_len | type | status | priority | label_count | component_count | customfield_count | link_count | flag_missing_ac | assignee_empty | storypoints_empty | epic | sprint |
|-----|-------------|----------------|------------|----------------|------|--------|---------|------------|----------------|------------------|-----------|----------------|---------------|------------------|------|--------|
| PROJ-123 | Cleaned summary | Cleaned desc | 45 | 320 | story | to do | high | 2 | 1 | 5 | 1 | 0 | 0 | 0 | PROJ-50 | Sprint 1 |

### Embedding Output

**embeddings.parquet:**

| key | embedding |
|-----|-----------|
| PROJ-123 | [0.123, -0.456, ..., 0.789] (384-D array) |

### Final Output

**clean_backlog.csv:**

| rank | key | score | type | status | priority | summary | epic | sprint |
|------|-----|-------|------|--------|---------|---------|------|--------|
| 1 | PROJ-124 | 2.87 | Bug | To Do | Critical | Fix... | PROJ-50 | Sprint 1 |
| 2 | PROJ-123 | 2.45 | Story | To Do | High | Impl... | PROJ-50 | Sprint 1 |

---

## Configuration Schema

### Structure

```yaml
validator:          # Backbone rules
  key_regex: ...
  date_formats: [...]
  required_fields: [...]
  optional_fields: [...]
  link_policy: strict
  csv_column_mapping: {...}

features:           # Feature extraction
  strip_html: true
  max_summary_len: 500
  categorical_fields: [type, status, priority]

embeddings:         # Embedding generation
  model_name: "..."
  model_path: "./models/..."
  dimension: 384
  device: cpu

faiss:              # FAISS indexing
  index_type: IVF-PQ
  nlist: 4096
  m: 16
  nprobe: 16

ranker:             # LambdaMART
  objective: lambdarank
  metric: ndcg
  num_leaves: 31
  learning_rate: 0.06
  n_estimators: 500
  group_by: epic

reranker:           # ColBERT (optional)
  enabled: false
  top_k: 50
  blend_weight: 0.7

guardrails:         # Post-ranking rules
  enabled: true
  rules: [...]

artifacts:          # Output paths
  base_dir: ./artifacts
  outputs: {...}

logging:            # Logging & benchmarking
  level: INFO
  track_performance: true
  benchmark_dir: ./benchmarks

targets:            # Quality targets
  max_processing_time_sec: 300
  min_ndcg_at_10: 0.85
```

### Extension Points

**Add new validator rules:**
```yaml
validator:
  required_fields:
    - your_custom_field
```

**Add custom feature:**
Edit `src/core/features.py` to extract from canonical dict.

**Add guardrail:**
```yaml
guardrails:
  rules:
    - name: "your_rule"
      condition: "..."
      action: "..."
```

---

## Performance Optimizations

### 1. Streaming Parsers
- XML: `iterparse` yields elements incrementally
- CSV: `read_csv(chunksize=...)` processes in batches
- Memory usage independent of dataset size

### 2. FAISS Index
- **IVF** (Inverted File): Coarse quantizer reduces search space
- **PQ** (Product Quantization): Compressed vectors (16× smaller)
- **nprobe**: Tunable recall/speed tradeoff

### 3. LightGBM
- Efficient gradient boosting (faster than XGBoost)
- Sparse feature support
- Early stopping on validation set

### 4. Artifact Caching
- Parquet format for DataFrames (columnar, compressed)
- Hash-based versioning for reproducibility
- Skip expensive stages if cached artifacts exist

---

## Testing Strategy

### Unit Tests

```python
# tests/test_validator.py
def test_validator_valid_key():
    validator = BackboneValidator(key_regex=r"^[A-Z]+-\d+$", ...)
    flags = validator._validate_row({"key": "PROJ-123", ...}, ...)
    assert flags["key_format_ok"] == True

def test_validator_invalid_key():
    flags = validator._validate_row({"key": "invalid", ...}, ...)
    assert flags["key_format_ok"] == False
```

### Integration Tests

```python
# tests/test_pipeline.py
def test_full_pipeline():
    pipeline = JIRAPipeline(config)
    result = pipeline.run("fixtures/test.xml")
    assert len(result) > 0
    assert "rank" in result.columns
```

### Benchmark Tests

```python
# tests/test_performance.py
@pytest.mark.benchmark
def test_validation_speed(benchmark):
    result = benchmark(validator.validate_file, "large_dataset.xml")
    assert benchmark.stats.stats.mean < 60.0  # < 60 seconds
```

---

## Security & Compliance

### 1. Local Processing
- No external API calls
- All processing happens on local machine
- No data leaves the system

### 2. Sensitive Data Handling
- HTML stripping removes potentially embedded scripts
- No raw text stored in FAISS (only embeddings)
- Redact credentials if needed (add to `utils/text.py`)

### 3. Versioning & Audit
- All artifacts versioned with hash
- `.meta.json` includes creation timestamp
- Logs track every stage execution

---

## Failure Modes & Recovery

### Validation Failure
**Symptom:** Pipeline stops with "missing required fields"

**Recovery:**
1. Check `artifacts/validation/backbone_report.csv`
2. Identify problematic rows
3. Fix source data or adjust `config/default.yaml`

### Embedding Failure
**Symptom:** OOM or slow performance

**Recovery:**
1. Reduce `batch_size` in config
2. Use GPU if available (`device: cuda`)
3. Truncate text fields (`max_summary_len`, `max_description_len`)

### FAISS Build Failure
**Symptom:** "nlist too large for dataset"

**Recovery:**
1. Reduce `nlist` (e.g., to `n_vectors // 10`)
2. Check embedding dimension matches config

### Ranking Instability
**Symptom:** Different order on same input

**Recovery:**
1. Use weak labels (heuristic) for consistency
2. Increase `min_data_in_leaf` in ranker config
3. Add seed to LightGBM params

---

## Future Enhancements

### 1. Multilingual Support
- Replace MiniLM with BGE-M3 (multilingual embeddings)
- Update config: `model_name: "BAAI/bge-m3"`

### 2. Real-time Inference API
- Add Flask/FastAPI wrapper around `pipeline.run()`
- Cache models in memory for fast inference

### 3. Active Learning
- Implement human-in-loop feedback collection
- Retrain LambdaMART with approved rankings

### 4. Distributed Processing
- Add Dask/Ray for multi-node scaling
- Shard dataset for parallel validation

### 5. Advanced Re-ranking
- Integrate full ColBERT-v2 (currently placeholder)
- Implement late-interaction scoring

---

## Maintenance

### Updating Dependencies

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Update specific package
pip install --upgrade lightgbm
```

### Model Updates

```bash
# Download new model version
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('model-name').save('models/model-files/new-model')"

# Update config
# config/default.yaml:
#   embeddings:
#     model_path: "./models/model-files/new-model"
```

### Config Migration

When adding new config fields:
1. Add to `config/default.yaml` with sensible defaults
2. Update factory functions (`create_validator`, etc.) to handle new fields
3. Add tests for new functionality

---

## Contact & Support

For issues, questions, or contributions:
- See `README.md` for quick start
- Check `logs/` for execution details
- Inspect `benchmarks/` for performance metrics
