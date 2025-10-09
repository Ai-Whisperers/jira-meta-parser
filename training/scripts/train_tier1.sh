#!/bin/bash
# Train LightGBM ranker on Tier 1 baseline dataset (602 issues)

set -e

echo "=========================================="
echo "TIER 1 TRAINING - Baseline (602 issues)"
echo "=========================================="

# Step 1: Generate artifacts if needed
if [ ! -d "artifacts/features" ] || [ ! -d "artifacts/labels" ]; then
    echo "Step 1: Generating artifacts..."
    python -m src.cli.dev full datasets/clean/tier1_baseline/dataset.csv
else
    echo "Step 1: Artifacts already exist, skipping generation"
fi

# Step 2: Train model
echo ""
echo "Step 2: Training LightGBM ranker..."
python -m training.train_ranker \
    --artifacts-dir ./artifacts \
    --output-dir ./training/model

# Step 3: Evaluate model
echo ""
echo "Step 3: Evaluating trained model..."
python -m training.evaluate evaluate \
    --model ./training/model/ltr_model.txt \
    --artifacts-dir ./artifacts \
    --output ./training/reports/tier1_eval.json

echo ""
echo "=========================================="
echo "✓ TIER 1 TRAINING COMPLETE!"
echo "=========================================="
echo "Model saved to: training/model/ltr_model.txt"
echo "Report saved to: training/reports/tier1_eval.json"
