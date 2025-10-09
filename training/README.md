# Training Module - LightGBM LambdaMART Ranker

This module contains all training-related functionality for the JIRA ticket ranker, separated from the inference pipeline for clean research and experimentation.

## Overview

The training module enables **actual machine learning training** of the LightGBM LambdaMART ranker using weak labels generated from the pipeline. Unlike the main pipeline (which uses heuristic scoring), this module trains an adaptive ML model that learns from data.

## Directory Structure

```
training/
├── __init__.py              # Module initialization
├── config.py                # Training-specific configuration
├── train_ranker.py          # Main training script
├── evaluate.py              # Evaluation utilities
├── utils.py                 # Helper functions
├── README.md                # This file
│
├── model/                   # Trained model artifacts
│   ├── ltr_model.txt        # LightGBM model file
│   ├── ltr_model.meta.json  # Feature names, config
│   └── ltr_model.encoders.pkl  # Categorical encoders
│
├── scripts/                 # Training experiments
│   ├── train_tier1.sh       # Train on 602 issues
│   ├── train_tier2.sh       # Train on 1,602 issues
│   └── train_tier3.sh       # Train on 745K issues
│
├── experiments/             # MLflow experiment tracking
│   └── mlruns/              # MLflow artifacts
│
├── reports/                 # Training reports
│   ├── training_report.json  # Metrics and analysis
│   └── comparison.json       # ML vs heuristic comparison
│
└── notebooks/               # Research notebooks
    ├── exploratory_analysis.ipynb
    ├── feature_importance.ipynb
    └── hyperparameter_tuning.ipynb
```

## Quick Start

### Prerequisites

1. Run the main pipeline to generate artifacts:
```bash
python -m src.cli.dev full datasets/clean/tier1_baseline/dataset.csv
```

This creates:
- `artifacts/features/variability_features_*.parquet`
- `artifacts/embeddings/embeddings_*.parquet`
- `artifacts/labels/weak_labels_*.csv`
- `artifacts/validation/backbone_report_*.csv`

### Training

Train the LightGBM ranker:

```bash
python -m training.train_ranker \
  --artifacts-dir ./artifacts \
  --output-dir ./training/model
```

**What this does:**
- Loads all preprocessed artifacts
- Prepares features using ranker.prepare_features()
- Splits data into train/val/test (70/15/15)
- **Actually trains LightGBM LambdaMART model**
- Saves trained model to `training/model/ltr_model.txt`
- Generates training report with metrics

### Evaluation

Evaluate the trained model:

```bash
python -m training.evaluate evaluate \
  --model ./training/model/ltr_model.txt \
  --artifacts-dir ./artifacts \
  --output ./training/reports/eval_results.json
```

### Generate Ranked Output

Use the trained model for ranking:

```bash
python -m training.evaluate rank \
  --model ./training/model/ltr_model.txt \
  --artifacts-dir ./artifacts \
  --output ./training/reports/ml_ranked_output.csv
```

## Training Configuration

Configuration is managed in `training/config.py` with these key parameters:

### Data Splits
```python
train_split: 0.7   # 70% for training
val_split: 0.15    # 15% for validation
test_split: 0.15   # 15% for testing
random_seed: 42    # Reproducibility
```

### Model Parameters (from config/default.yaml)
```yaml
ranker:
  objective: "lambdarank"
  metric: "ndcg"
  num_leaves: 31
  learning_rate: 0.06
  n_estimators: 500
  eval_at: [10, 20, 50]
```

### Cross-Validation
```python
cv_folds: 5
cv_strategy: "group"  # Group by epic/sprint
```

## What Makes This Different from Main Pipeline?

| Aspect | Main Pipeline (src/pipeline.py) | Training Module (training/) |
|--------|--------------------------------|----------------------------|
| **Purpose** | Inference only | Training + experimentation |
| **Ranking Method** | Heuristic formula | **ML model (LightGBM)** |
| **Uses Weak Labels** | ❌ No (generates but ignores) | ✅ Yes (for training) |
| **Model Training** | ❌ Never happens | ✅ **Actual training** |
| **Saved Artifacts** | CSV outputs only | **Trained model files** |
| **Evaluation** | None | Train/val/test metrics |
| **Adaptability** | Fixed weights | **Learns from data** |

## Training Process

### Step 1: Data Loading
```python
# Load preprocessed artifacts
features_df = pd.read_parquet("artifacts/features/variability_features_*.parquet")
embeddings_df = pd.read_parquet("artifacts/embeddings/embeddings_*.parquet")
weak_labels_df = pd.read_csv("artifacts/labels/weak_labels_*.csv")
validation_df = pd.read_csv("artifacts/validation/backbone_report_*.csv")
```

### Step 2: Feature Preparation
```python
# Prepare features (same as ranker would use for inference)
X, groups = ranker.prepare_features(features_df, embeddings_df, validation_df)
# X shape: (n_samples, n_features)
# Features: text lengths, counts, flags, embeddings, hygiene scores, etc.
```

### Step 3: Label Extraction
```python
# Use weak labels as training targets
y = weak_labels_df["relevance_score"].values
# y values: 0.45 - 3.64 (continuous relevance scores)
```

### Step 4: Data Splitting
```python
# Split into train/val/test sets
train_idx, val_idx, test_idx = split_data(X, y, groups)
# Train: 70%, Val: 15%, Test: 15%
# Respects group structure (epics/sprints)
```

### Step 5: Model Training
```python
# Train LightGBM LambdaMART model
metrics = ranker.train(
    X_train,
    y_train,
    groups_train,
    eval_set=(X_val, y_val, groups_val)
)
# Uses gradient boosting with LambdaMART loss
# Optimizes for NDCG (Normalized Discounted Cumulative Gain)
```

### Step 6: Model Saving
```python
# Save trained model
ranker.save("training/model/ltr_model.txt")
# Saves:
# - ltr_model.txt (LightGBM model)
# - ltr_model.meta.json (feature names, config)
# - ltr_model.encoders.pkl (categorical encoders)
```

### Step 7: Evaluation
```python
# Evaluate on test set
test_metrics = ranker.predict(X_test)
metrics = compute_ranking_metrics(y_test, y_pred, groups_test)
# Metrics: NDCG@10, NDCG@20, NDCG@50, Spearman, Kendall's Tau
```

## Expected Outputs

### Training Report (training/reports/training_report.json)
```json
{
  "config": {...},
  "data": {
    "train_size": 421,
    "val_size": 90,
    "test_size": 91,
    "n_features": 15,
    "feature_names": [
      "summary_len", "description_len", "label_count",
      "component_count", "customfield_count", "link_count",
      "flag_missing_ac", "assignee_empty", "storypoints_empty",
      "type_encoded", "status_encoded", "priority_encoded",
      "required_ok", "dates_ok", "embedding_norm"
    ]
  },
  "metrics": {
    "train": {
      "ndcg@10": 0.8523,
      "ndcg@20": 0.8691,
      "ndcg": 0.8845,
      "spearman": 0.7234,
      "kendall_tau": 0.6012
    },
    "val": {
      "ndcg@10": 0.8102,
      "ndcg@20": 0.8289,
      "ndcg": 0.8512,
      "spearman": 0.6823,
      "kendall_tau": 0.5634
    },
    "test": {
      "ndcg@10": 0.8156,
      "ndcg@20": 0.8334,
      "ndcg": 0.8567,
      "spearman": 0.6891,
      "kendall_tau": 0.5702
    }
  }
}
```

### Trained Model Files
- `training/model/ltr_model.txt` - LightGBM model (binary format)
- `training/model/ltr_model.meta.json` - Metadata
- `training/model/ltr_model.encoders.pkl` - Categorical encoders

## Comparison with Heuristic Baseline

The heuristic baseline uses:
```python
score = 0.4 * priority_score + 0.3 * status_score + 0.3 * hygiene_score
```

Expected improvements with trained model:
- **NDCG@10**: +5-10% improvement
- **Spearman**: +10-15% improvement
- **Adaptability**: Learns patterns from data vs fixed rules

## Progressive Training Strategy

Train on increasingly larger datasets:

### Tier 1 (Baseline - 602 issues)
```bash
python -m training.train_ranker --artifacts-dir ./artifacts
```
- **Purpose**: Validate training works
- **Time**: ~30 seconds
- **Model size**: ~50KB

### Tier 2 (Merged - 1,602 issues)
```bash
# First generate Tier 2 artifacts
python -m src.cli.dev full datasets/clean/tier2_merged/dataset.csv

# Then train
python -m training.train_ranker --artifacts-dir ./artifacts
```
- **Purpose**: Improve model with more data
- **Time**: ~1 minute
- **Expected improvement**: +2-5% NDCG@10

### Tier 3 (Production - 745K issues)
```bash
# Note: Tier 3 currently has data quality issues (3 missing required fields)
# Clean dataset first, then:
python -m src.cli.dev full datasets/clean/tier3_production_clean/dataset.csv
python -m training.train_ranker --artifacts-dir ./artifacts
```
- **Purpose**: Production-scale model
- **Time**: ~10-20 minutes
- **Expected improvement**: +10-15% NDCG@10

## Advanced Features

### Hyperparameter Tuning
```python
# TODO: Implement with Optuna
python -m training.hyperparameter_search \
  --n-trials 50 \
  --timeout 3600
```

### Feature Importance Analysis
```python
# TODO: Implement
python -m training.feature_analysis \
  --model ./training/model/ltr_model.txt
```

### Cross-Validation
```python
# TODO: Implement
python -m training.cross_validate \
  --cv-folds 5 \
  --strategy group
```

## Troubleshooting

### Error: "Could not find all required artifacts"
**Solution**: Run the main pipeline first:
```bash
python -m src.cli.dev full datasets/clean/tier1_baseline/dataset.csv
```

### Error: "Model not trained. Call train() first."
**Solution**: Train the model before evaluating:
```bash
python -m training.train_ranker
```

### Warning: "Group size too small"
**Solution**: Ensure dataset has enough samples per group (epic/sprint). Consider using `group_by: null` in config.

## Next Steps

1. ✅ **Implement training** - Done! Use `train_ranker.py`
2. ✅ **Evaluate model** - Done! Use `evaluate.py`
3. ⏳ **Compare with heuristic** - TODO: Create comparison script
4. ⏳ **Hyperparameter tuning** - TODO: Add Optuna integration
5. ⏳ **Feature analysis** - TODO: Add feature importance plots
6. ⏳ **Production deployment** - TODO: Update main pipeline to use trained model

## Contributing

When adding new training features:
1. Add configuration to `config.py`
2. Implement in separate module
3. Add CLI command to `scripts/`
4. Document in this README
5. Add tests to `tests/training/`

## References

- [LightGBM LambdaMART Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambdarank)
- [Learning to Rank (Microsoft Research)](https://www.microsoft.com/en-us/research/project/mslr/)
- [NDCG Metric Explanation](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)

---

**Status**: ✅ Training module ready for use
**Last Updated**: 2025-10-09
**Author**: Claude Code
