# Quick Start Guide

**Get your JIRA backlog ranked in 5 minutes.**

---

## Prerequisites

- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- JIRA export file (XML or CSV)

---

## Installation

```bash
# Clone repository
cd jira-ticket-meta-parser

# Install dependencies
pip install -r requirements.txt
```

**That's it.** Pre-downloaded models are already in `models/`.

---

## Usage

### Option 1: Production CLI (Simplest)

```bash
python -m src.cli.prod --input datasets/JIRA.xml --output clean_backlog.csv --verbose
```

**Output:**
```
JIRA Ticket Meta Parser v1.0.0
==================================================

[1/4] Loading pipeline...
[2/4] Processing datasets/JIRA.xml...
[3/4] Saving to clean_backlog.csv...
[4/4] Complete!

✓ Successfully ranked 600 issues
  Output: clean_backlog.csv

Top 10 issues:
rank  key        score  type   priority
1     PROJ-124   2.87   Bug    Critical
2     PROJ-123   2.45   Story  High
...
```

### Option 2: Developer CLI (Stage-by-Stage)

```bash
# Step 1: Validate
python -m src.cli.dev validate datasets/JIRA.xml

# Step 2: Extract features
python -m src.cli.dev extract datasets/JIRA.xml

# Step 3: Generate embeddings
python -m src.cli.dev embed

# Step 4: Build FAISS index
python -m src.cli.dev index

# Step 5: Run full pipeline
python -m src.cli.dev full datasets/JIRA.xml
```

**Check status:**
```bash
python -m src.cli.dev status
```

**Output:**
```
Pipeline Status
==================================================

VALIDATION:
  - backbone_report_a1b2c3d4.csv (45.2 KB)
  - backbone_summary_a1b2c3d4.json (1.3 KB)

FEATURES:
  - variability_features_e5f6g7h8.parquet (120.5 KB)

EMBEDDINGS:
  - embeddings_i9j0k1l2.parquet (890.3 KB)
...
```

---

## Understanding the Output

### Final Backlog (`clean_backlog.csv`)

| Column | Description |
|--------|-------------|
| `rank` | Final rank (1 = highest priority) |
| `key` | JIRA issue key |
| `score` | Ranking score (higher = more important) |
| `type` | Issue type (Story, Bug, Task, etc.) |
| `status` | Current status |
| `priority` | JIRA priority |
| `summary` | Issue summary |
| `epic` | Parent epic (if applicable) |
| `sprint` | Target sprint |

**Use this CSV to:**
- Import back into JIRA
- Share with stakeholders
- Drive sprint planning

---

## Configuration (Optional)

Edit `config/default.yaml` to customize:

### ⚡ FIRST: Optimize for Your Data

**IMPORTANT:** The default config is tuned for the included 602-issue dataset. For your own data:

**Step 1 - Analyze your data distribution:**
```bash
# Check priority/status/type distributions
python << 'EOF'
import xml.etree.ElementTree as ET
from collections import Counter

tree = ET.parse('your_export.xml')
items = tree.getroot().findall('.//item') or tree.getroot().findall('.//issue')

priorities = Counter(item.findtext('priority', '').strip() for item in items)
statuses = Counter(item.findtext('status', '').strip() for item in items)

print("Priorities:", dict(priorities))
print("Statuses:", dict(statuses))
EOF
```

**Step 2 - Adjust weak labeling weights:**
```yaml
ranker:
  weak_labels:
    # If one field is uniform (90%+ same value) → reduce weight
    # If one field has good spread → increase weight
    priority_weight: 0.2  # Example: 90% Medium priority
    status_weight: 0.4    # Example: Good status diversity
    hygiene_weight: 0.4   # Usually has best variance
```

**Step 3 - Map ALL your status values:**
```yaml
ranker:
  weak_labels:
    status_scores:
      # Add EVERY status from your export!
      "your_status_1": 4.0
      "your_status_2": 3.0
      # ... etc
```

**See `local-reports/DATA_ANALYSIS_REPORT.md` for full optimization guide.**

---

### Adjust for GPU (Faster)

```yaml
embeddings:
  device: "cuda"  # Change from "cpu"

faiss:
  use_gpu: true
```

**Requirement:** Install `faiss-gpu` instead of `faiss-cpu`.

### Adjust for Small Datasets (<1000 issues)

```yaml
faiss:
  nlist: 256  # Reduce from 4096
```

### Enable Re-ranking (Better Quality)

```yaml
reranker:
  enabled: true
  top_k: 50
```

### Enable Data Augmentation

```yaml
preprocessing:
  enabled: true
  augmentation_factor: 2  # 2x-5x dataset size
```

**Use with:** `export AUGMENTATION_FACTOR=3`

---

## Troubleshooting

### 1. Validation fails

**Error:**
```
✗ Error: Validation failed. See backbone_report.csv for details.
```

**Fix:**
1. Open `artifacts/validation/backbone_report.csv`
2. Look for rows where `required_ok = False`
3. Fix missing fields in your JIRA export

### 2. Out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Fix:**
```yaml
# config/default.yaml
embeddings:
  batch_size: 16  # Reduce from 32
  device: "cpu"   # Use CPU instead of GPU
```

### 3. Import error

**Error:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Fix:**
```bash
pip install -r requirements.txt
```

---

## Next Steps

### 1. Inspect Results

```bash
# View top 20 ranked issues
head -20 clean_backlog.csv
```

### 2. Check Performance

```bash
# View benchmark data
cat benchmarks/benchmarks_*.json
```

**Example output:**
```json
{
  "validation": [
    {"timestamp": "2025-10-04T12:00:00", "duration_sec": 2.34, "status": "success"}
  ],
  "embeddings": [
    {"timestamp": "2025-10-04T12:00:05", "duration_sec": 45.67, "status": "success"}
  ]
}
```

### 3. Train with Real Labels (Advanced)

Once you have **labeled data** (PM-approved rankings):

```python
from src.pipeline import JIRAPipeline
from src.utils import Config

config = Config()
pipeline = JIRAPipeline(config.to_dict())

# Load your labels
labels = pd.read_csv("labels.csv")  # key, relevance_score

# Merge with features
features_df = pipeline._load_cached("features", "variability_features")
embeddings_df = pipeline._load_cached("embeddings", "embeddings")
validation_df = pipeline._load_cached("validation", "backbone_report")

df = features_df.merge(labels, on="key")
X, groups = pipeline.ranker.prepare_features(features_df, embeddings_df, validation_df)
y = df["relevance_score"].values

# Train
metrics = pipeline.ranker.train(X, y, groups)
print(f"NDCG@10: {metrics['ndcg@10']}")

# Save trained model
pipeline.ranker.save("artifacts/models/ltr_model.txt")
```

---

## Common Workflows

### Workflow 1: Weekly Backlog Refresh

```bash
# Export fresh JIRA data (via JIRA UI)
# Run pipeline
python -m src.cli.prod -i weekly_export.xml -o backlog_$(date +%Y%m%d).csv -v

# Review output
head backlog_*.csv
```

### Workflow 2: A/B Testing Different Configs

```bash
# Create custom config
cp config/default.yaml config/experiment.yaml

# Edit experiment.yaml (change ranker params)

# Run with custom config
python -m src.cli.dev full datasets/JIRA.xml --config config/experiment.yaml

# Compare outputs
diff artifacts/backlogs/clean_backlog_*.csv
```

### Workflow 3: Debugging Specific Issues

```bash
# Validate only
python -m src.cli.dev validate datasets/JIRA.xml

# Check report
cat artifacts/validation/backbone_report.csv | grep "PROJ-123"

# Extract features for one issue
python -m src.cli.dev extract datasets/JIRA.xml
cat artifacts/features/variability_features_*.parquet | grep "PROJ-123"
```

---

## Tips & Best Practices

### 1. Always Validate First

```bash
# Run validation before full pipeline
python -m src.cli.dev validate your_export.xml
```

Catches data quality issues early.

### 2. Use Version Control for Configs

```bash
git add config/default.yaml
git commit -m "Update ranker hyperparameters"
```

Ensures reproducibility.

### 3. Archive Outputs

```bash
# Create timestamped archive
tar -czf artifacts_$(date +%Y%m%d).tar.gz artifacts/
```

### 4. Monitor Performance

```bash
# Check if pipeline meets targets
cat benchmarks/benchmarks_*.json | jq '.[] | select(.duration_sec > 300)'
```

If any stage exceeds 5 minutes, optimize config.

---

## Resources

- **Full Documentation**: See `README.md`
- **Architecture Details**: See `ARCHITECTURE.md`
- **Context Documents**:
  - `metacontext.md`: Full pipeline specification
  - `validator-context.md`: Validator architecture

---

## Support

**Need help?**

1. Check logs: `logs/pipeline.log`
2. Review benchmarks: `benchmarks/*.json`
3. Inspect artifacts: `python -m src.cli.dev status`

**Found a bug?** Open an issue with:
- Input file (or sample)
- Config used
- Full error message from logs
