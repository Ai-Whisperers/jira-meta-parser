# Dataset Merge & Enhancement Strategy

**Generated:** 2025-10-06
**Status:** 📋 **READY FOR IMPLEMENTATION**

---

## Executive Summary

Analysis of available datasets reveals **GFG_FINAL.csv** as the optimal merge candidate with the reference dataset, providing a **81x increase** in training data (602 → 49,602 issues) while maintaining schema compatibility and data quality.

**Key Recommendation:** Merge GFG_FINAL.csv with JIRA.csv using column harmonization strategy outlined below.

---

## Dataset Inventory

### 1. Reference Dataset (datasets/JIRA.csv)
- **Size:** 602 issues
- **Columns:** 166
- **Format:** Jira Cloud export (extensive custom fields)
- **Quality:** Excellent
  - Required fields: 100% coverage
  - Priority: 100% | Assignee: 77% | Description: 77%
- **Use Case:** Training baseline, schema reference

### 2. GFG_FINAL.csv (merge/CSV_JIRA_ARCHIVE/)
- **Size:** 49,000 issues
- **Columns:** 491 (70 overlap with reference)
- **Format:** Similar Jira export (likely Atlassian products)
- **Quality:** Good
  - Required fields: 85-100% coverage
  - Priority: 85% | Assignee: 8% | Description: 100%
- **Use Case:** 🎯 **PRIMARY MERGE TARGET**

### 3. RedHat Datasets (merge/RedHat_OpenJira_ProcessDataset2024_Inputs/)
- **Size:** ~505K issues (251 CSV files)
- **Columns:** 9 (minimal schema)
- **Format:** Simplified export
- **Quality:** Poor for ML training
  - Has: Issue key, Type, Status, Project info, Dates
  - Missing: Priority, Summary, Description, Assignee, Reporter
- **Use Case:** ❌ **NOT RECOMMENDED** (insufficient for weak labeling)

### 4. ThePublicJiraDataset (merge/ThePublicJiraDataset/)
- **Size:** 2.7M issues
- **Format:** MongoDB dump (requires extraction)
- **Quality:** Unknown (needs MongoDB analysis)
- **Use Case:** ⏳ **FUTURE CONSIDERATION** (requires MongoDB setup)

---

## Training Data Requirements

### Critical Fields (from config/default.yaml)

**Required (fail-fast if missing):**
- ✅ key, summary, type, status, priority, created, updated

**Important for Weak Labeling:**
- ✅ priority (weight: 0.2) - Needs 85%+ coverage
- ✅ status (weight: 0.4) - Needs comprehensive mapping
- ✅ hygiene factors (weight: 0.4) - Assignee, description, story_points

**Important for Features:**
- ✅ description - For text embeddings (384-D vectors)
- ✅ components, labels - For categorical features
- ✅ parent, epic_link - For grouping/ranking

### Dataset Compatibility Assessment

| Field | Reference | GFG_FINAL | RedHat | Required? |
|-------|-----------|-----------|--------|-----------|
| key | ✅ 100% | ✅ 100% | ✅ 100% | YES |
| summary | ✅ 100% | ✅ 100% | ❌ Missing | YES |
| type | ✅ 100% | ✅ 100% | ✅ 100% | YES |
| status | ✅ 100% | ✅ 100% | ✅ 100% | YES |
| priority | ✅ 100% | ✅ 85% | ❌ Missing | YES |
| created | ✅ 100% | ✅ 100% | ✅ 100% | YES |
| updated | ✅ 100% | ✅ 100% | ❌ Missing | YES |
| description | ✅ 77% | ✅ 100% | ❌ Missing | Important |
| assignee | ✅ 77% | ⚠️ 8% | ❌ Missing | Important |

**Verdict:**
- ✅ GFG_FINAL: Compatible (all required fields present)
- ❌ RedHat: Incompatible (missing 4/7 required fields)

---

## Recommended Strategy: GFG_FINAL Merge

### Phase 1: Column Harmonization

**Approach:** Map GFG columns to reference schema using `csv_column_mapping` in config.

**Mapping Plan:**
```yaml
# Add to config/default.yaml → validator.csv_column_mapping
csv_column_mapping:
  # Core fields (both datasets)
  "Issue key": "key"
  "Summary": "summary"
  "Issue Type": "type"
  "Status": "status"
  "Priority": "priority"
  "Created": "created"
  "Updated": "updated"

  # Additional fields (may vary)
  "Assignee": "assignee"
  "Reporter": "reporter"
  "Description": "description"
  "Component/s": "components"
  "Labels": "labels"

  # Custom fields (check GFG schema)
  "Parent": "parent"  # GFG may use different name
  # Add GFG-specific mappings as needed
```

**Action Items:**
1. Compare GFG column names for custom fields
2. Update config mapping to handle both schemas
3. Validate column alignment with test run

### Phase 2: Data Cleaning

**Priority Standardization:**
- GFG has 85% priority coverage (14% missing)
- Options:
  1. **Impute missing** → Set to "Medium" (matches reference 90% distribution)
  2. **Drop rows** → Reduces dataset to 41.7K (still 69x increase)
  3. **Keep as-is** → Let weak labeler handle nulls (defaults to 2.0 score)

**Recommendation:** Option 1 (Impute to "Medium") - maintains dataset size while providing reasonable default.

**Status Mapping:**
- Reference uses: backlog, in progress, done, user testing, etc.
- GFG uses: Needs Triage, Closed, Open, etc.
- **Action:** Analyze GFG status values and extend `status_scores` mapping in config

```bash
# Check GFG status distribution
python -c "
import pandas as pd
df = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')
print(df['Status'].value_counts())
"
```

**Date Format Alignment:**
- Reference: RFC-822 format ("Thu, 18 Sep 2025 14:36:50 -0600")
- GFG: Check format and ensure in `date_formats` list (config line 10-14)

### Phase 3: Deduplication

**Check for overlaps:**
```bash
# Find duplicate issue keys between datasets
python -c "
import pandas as pd
ref = pd.read_csv('datasets/JIRA.csv')
gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')
overlap = set(ref['Issue key']) & set(gfg['Issue key'])
print(f'Overlapping issues: {len(overlap)}')
if overlap:
    print('Sample:', list(overlap)[:5])
"
```

**Strategy:**
- If overlaps exist: Keep reference version (higher quality metadata)
- If no overlaps: Simple concatenation

### Phase 4: Merge Execution

**Option A: Direct CSV Merge (Simple)**
```python
import pandas as pd

# Load datasets
ref = pd.read_csv('datasets/JIRA.csv')
gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')

# Standardize columns (use common subset or harmonize names)
common_cols = list(set(ref.columns) & set(gfg.columns))

# Optional: Impute missing priorities
gfg['Priority'] = gfg['Priority'].fillna('Medium')

# Concatenate
merged = pd.concat([ref[common_cols], gfg[common_cols]], ignore_index=True)

# Remove duplicates (keep first = reference)
merged = merged.drop_duplicates(subset='Issue key', keep='first')

# Save
merged.to_csv('datasets/JIRA_MERGED.csv', index=False)
print(f'Merged dataset: {len(merged)} issues')
```

**Option B: Pipeline-Based Merge (Robust)**
```bash
# 1. Validate GFG with current config
python -m src.cli.dev validate datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv

# 2. Check validation report
cat artifacts/validation/backbone_report.csv | head -20

# 3. If issues found: Update config mappings

# 4. Merge after successful validation
python merge_datasets.py  # Create custom merge script
```

### Phase 5: Post-Merge Validation

**Quality Checks:**
1. **Row count:** Should be ~49,602 (602 + 49,000)
2. **Required fields:** 100% coverage on key, summary, type, status, created
3. **Priority distribution:** Check for reasonable variance
4. **Status coverage:** Ensure all values mapped in config
5. **Weak label distribution:** Should have wide range (0.5-3.5)

**Validation Commands:**
```bash
# Run validation on merged dataset
python -m src.cli.dev validate datasets/JIRA_MERGED.csv

# Generate weak labels and check distribution
python -m src.cli.dev full datasets/JIRA_MERGED.csv
cat artifacts/labels/weak_labels.csv | python -c "
import pandas as pd
import sys
df = pd.read_csv(sys.stdin)
print(df['relevance_score'].describe())
"
```

---

## Alternative Strategies (Not Recommended)

### RedHat Integration
**Why not:** Missing 4/7 required fields (Priority, Summary, Description, Updated)

**Potential use cases:**
- Status transition analysis (has Created + Resolved dates)
- Project type distribution study
- Issue type frequency analysis

**If absolutely needed:**
- Could be used for **data augmentation** by generating synthetic summaries/priorities
- Requires significant preprocessing effort (not worth it given GFG availability)

### ThePublicJiraDataset Extraction
**Why not yet:** Requires MongoDB setup and schema analysis

**Future consideration:**
```bash
# Would require:
1. Install MongoDB
2. Import dump: mongorestore --archive=mongodump-JiraReposAnon.archive
3. Export to CSV with proper schema
4. Apply same merge strategy as GFG
```

**Benefit:** 2.7M issues (massive scale)
**Cost:** High extraction effort, unknown schema compatibility

---

## Implementation Checklist

### Pre-Merge (Configuration)
- [ ] Analyze GFG status values and extend status_scores mapping
- [ ] Verify GFG date formats match config allowlist
- [ ] Check for custom field mappings (Parent, Epic, etc.)
- [ ] Update csv_column_mapping in config/default.yaml

### Merge Execution
- [ ] Run deduplication check
- [ ] Impute missing GFG priorities (if using Option 1)
- [ ] Execute merge script (Option A or B)
- [ ] Validate merged dataset schema

### Post-Merge Validation
- [ ] Run pipeline validation stage
- [ ] Check weak label distribution (should be 0.5-3.5 range)
- [ ] Verify FAISS index build (may need to increase nlist for larger dataset)
- [ ] Test full pipeline run
- [ ] Compare NDCG@10 metrics (should improve with more data)

### Production Readiness
- [ ] Update README.md with new dataset info
- [ ] Document GFG-specific mappings in CLAUDE.md
- [ ] Archive original datasets (keep merge/CSV_JIRA_ARCHIVE untouched)
- [ ] Commit JIRA_MERGED.csv as new baseline

---

## Expected Training Improvements

### Dataset Size Impact
- **Before:** 602 issues
- **After:** ~49,602 issues (81x increase)
- **Benefit:** Better model generalization, reduced overfitting

### Feature Diversity
- **Description coverage:** 77% → 92% (more text data for embeddings)
- **Priority variance:** Low (90% Medium) → Potentially higher (need to analyze GFG distribution)
- **Status variety:** Both datasets contribute different status workflows

### Model Performance Predictions
- **NDCG@10:** Expected improvement from 0.85 → 0.90+ (more training examples)
- **Recall@100:** Better coverage of edge cases
- **Generalization:** Handles diverse Jira export formats

### FAISS Index Considerations
- **Current config:** nlist=4096 (optimized for small dataset)
- **Recommendation:** Increase to 8192 for 50K dataset
- **GPU consideration:** Processing 50K embeddings may benefit from faiss-gpu

```yaml
# Update config/default.yaml → faiss section
faiss:
  nlist: 8192  # Increased from 4096
  nprobe: 32   # Increased from 16 for better recall
```

---

## Next Steps

1. **Immediate (Today):**
   ```bash
   # Analyze GFG status values
   python -c "import pandas as pd; df = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv'); print(df['Status'].value_counts())"

   # Check for duplicates
   python -c "import pandas as pd; ref = pd.read_csv('datasets/JIRA.csv'); gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv'); print(f'Overlaps: {len(set(ref[\"Issue key\"]) & set(gfg[\"Issue key\"]))}')"
   ```

2. **Short-term (This Week):**
   - Update config mappings based on GFG schema
   - Create merge script (use Option A for simplicity)
   - Run post-merge validation

3. **Medium-term (Next Sprint):**
   - Train model on merged dataset
   - Compare performance metrics
   - Consider ThePublicJiraDataset extraction if need >1M issues

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GFG schema incompatibility | High | Pre-validate with test run before merge |
| Status value unmapped | Medium | Extend status_scores mapping comprehensively |
| Date format mismatch | Medium | Add GFG format to date_formats list |
| FAISS index build failure | Low | Increase nlist, reduce nprobe if needed |
| Degraded model performance | Low | Keep reference dataset separate for A/B testing |

---

## Success Criteria

✅ Merged dataset passes validation (100% required fields)
✅ Weak label distribution shows good variance (std > 0.5)
✅ FAISS index builds successfully
✅ Full pipeline runs without errors
✅ NDCG@10 ≥ 0.85 (maintain or improve from baseline)
✅ PM approval@20 ≥ 80% on test queries

---

## Appendix: Quick Start Commands

```bash
# 1. Analyze GFG dataset
python -c "
import pandas as pd
gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')
print(f'Rows: {len(gfg):,}')
print(f'\nStatus distribution:')
print(gfg['Status'].value_counts())
print(f'\nPriority distribution:')
print(gfg['Priority'].value_counts(dropna=False))
print(f'\nDate format sample:')
print(gfg['Created'].head(3))
"

# 2. Create merge script
cat > merge_datasets.py << 'EOF'
import pandas as pd

# Load datasets
ref = pd.read_csv('datasets/JIRA.csv')
gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')

# Get common columns
common_cols = sorted(list(set(ref.columns) & set(gfg.columns)))
print(f"Common columns: {len(common_cols)}")

# Impute missing priorities
gfg['Priority'] = gfg['Priority'].fillna('Medium')

# Merge
merged = pd.concat([ref[common_cols], gfg[common_cols]], ignore_index=True)
merged = merged.drop_duplicates(subset='Issue key', keep='first')

# Save
merged.to_csv('datasets/JIRA_MERGED.csv', index=False)
print(f"Merged dataset: {len(merged):,} issues")
print(f"Columns: {len(merged.columns)}")
EOF

# 3. Execute merge
python merge_datasets.py

# 4. Validate merged dataset
python -m src.cli.dev validate datasets/JIRA_MERGED.csv

# 5. Run full pipeline
python -m src.cli.dev full datasets/JIRA_MERGED.csv
```

---

**Document Owner:** Claude Code
**Last Updated:** 2025-10-06
**Next Review:** After merge execution
