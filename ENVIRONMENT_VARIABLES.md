# Environment Variables - Human-in-the-Loop Controls

This document describes environment variables for controlling pipeline behavior with human oversight.

---

## Weak Labeling Control

### `WEAK_LABEL_APPROVAL`

Controls human approval workflow for generated weak labels.

**Values:**
- `auto` (default) - Automatically approve generated labels
- `manual` - Require human review and approval

**Usage:**
```bash
# Auto mode (no prompts)
export WEAK_LABEL_APPROVAL=auto
python -m src.cli.dev full datasets/JIRA.xml

# Manual mode (human review required)
export WEAK_LABEL_APPROVAL=manual
python -m src.cli.dev full datasets/JIRA.xml
```

**Manual Mode Workflow:**
1. Pipeline generates weak labels
2. Shows score distribution statistics
3. Displays top 10 and bottom 10 ranked issues
4. Prompts: "Approve these labels? (yes/no/adjust)"
   - `yes` - Continue with training
   - `no` - Abort pipeline
   - `adjust` - Modify weights and regenerate

---

## Data Augmentation Control

### `AUGMENTATION_FACTOR`

Controls how many synthetic variations to create per original issue.

**Values:**
- `1` - No augmentation (original data only)
- `2` - 2x dataset size (default)
- `3` - 3x dataset size
- `4` - 4x dataset size
- `5` - 5x dataset size (maximum)

**Usage:**
```bash
# Default 2x augmentation
export AUGMENTATION_FACTOR=2

# No augmentation
export AUGMENTATION_FACTOR=1

# Maximum 5x augmentation
export AUGMENTATION_FACTOR=5
```

**Example Output:**
```
Original: 602 issues
Augmented: 1,806 issues (3x)
```

---

### `HUMAN_REVIEW_MODE`

Enables human approval before generating synthetic variations.

**Values:**
- `false` (default) - Automatic augmentation
- `true` - Require human approval

**Usage:**
```bash
# Enable human review for augmentation
export HUMAN_REVIEW_MODE=true
python -m src.cli.dev full datasets/JIRA.xml
```

**Human Review Workflow:**
1. Shows augmentation plan:
   - Original dataset size
   - Augmentation factor
   - Resulting total size
   - Enabled techniques (text perturbation, priority shuffle, status variation)
2. Prompts: "Proceed with augmentation? (yes/no)"
   - `yes` - Generate variations
   - `no` - Skip augmentation, use original data only

---

## Combined Example Workflows

### Workflow 1: Fully Automated (Default)

```bash
# No environment variables needed
python -m src.cli.dev full datasets/JIRA.xml
```

**Result:**
- Weak labels auto-generated (priority 40%, status 30%, hygiene 30%)
- No augmentation (preprocessing.enabled: false in config)
- Pipeline runs end-to-end without prompts

---

### Workflow 2: Human-Reviewed Weak Labels

```bash
export WEAK_LABEL_APPROVAL=manual

python -m src.cli.dev full datasets/JIRA.xml
```

**Result:**
- Pipeline pauses after weak label generation
- Shows label statistics and top/bottom issues
- Waits for human approval before continuing
- Can adjust weights if labels look incorrect

---

### Workflow 3: Human-Reviewed Augmentation

```bash
export HUMAN_REVIEW_MODE=true
export AUGMENTATION_FACTOR=3

python -m src.cli.dev full datasets/JIRA.xml
```

**Config requirement:**
```yaml
# config/default.yaml
preprocessing:
  enabled: true  # Must be enabled
```

**Result:**
- Weak labels auto-generated
- Augmentation plan displayed (3x = 1,806 total issues)
- Waits for human approval
- If approved, generates 3 variations per original issue

---

### Workflow 4: Full Human Control

```bash
export WEAK_LABEL_APPROVAL=manual
export HUMAN_REVIEW_MODE=true
export AUGMENTATION_FACTOR=4

python -m src.cli.dev full datasets/JIRA.xml
```

**Config requirement:**
```yaml
preprocessing:
  enabled: true
```

**Result:**
- Two human checkpoints:
  1. Approve weak labels (with weight adjustment option)
  2. Approve 4x augmentation plan
- Maximum control over training data quality

---

## Configuration File vs Environment Variables

**Environment Variables** (this file):
- Runtime overrides
- Human-in-the-loop controls
- Session-specific behavior

**config/default.yaml**:
- Default pipeline settings
- Algorithmic parameters
- Persistent configuration

**Precedence:** Environment variables override config file settings where applicable.

---

## Best Practices

### For Initial Training

```bash
# Review everything first time
export WEAK_LABEL_APPROVAL=manual
export HUMAN_REVIEW_MODE=true
export AUGMENTATION_FACTOR=2

python -m src.cli.dev full datasets/JIRA.xml
```

**Why:**
- Understand weak label distribution
- Verify augmentation doesn't create nonsense
- Adjust weights if needed

---

### For Production Re-Training

```bash
# Automated with known-good settings
export WEAK_LABEL_APPROVAL=auto
export AUGMENTATION_FACTOR=2

python -m src.cli.dev full datasets/JIRA.xml
```

**Why:**
- Faster iteration
- Use approved configuration
- No manual intervention needed

---

### For Experimentation

```bash
# Try different augmentation levels
for factor in 1 2 3 4 5; do
    export AUGMENTATION_FACTOR=$factor
    python -m src.cli.dev full datasets/JIRA.xml
    # Compare model metrics
done
```

**Why:**
- Find optimal augmentation level
- Compare NDCG@10, Kendall τ across runs
- Automated A/B testing

---

## Troubleshooting

### Issue: Weak labels look wrong (all same score)

**Solution:**
```bash
export WEAK_LABEL_APPROVAL=manual
# Run pipeline, choose "adjust" when prompted
# Increase hygiene_weight or priority_weight
```

---

### Issue: Augmentation creates too much data

**Solution:**
```bash
# Reduce factor
export AUGMENTATION_FACTOR=1  # No augmentation
# Or disable in config
# preprocessing.enabled: false
```

---

### Issue: Want to skip weak labeling entirely

**Solution:**
```yaml
# config/default.yaml
ranker:
  weak_labels:
    enabled: false
```

Then provide real labels via training script.

---

## See Also

- `config/default.yaml` - Full configuration reference
- `README.md` - Pipeline overview
- `QUICKSTART.md` - Getting started guide
- `src/core/weak_labeler.py` - Weak labeling implementation
- `src/core/preprocessor.py` - Augmentation implementation
