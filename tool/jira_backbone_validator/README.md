
# jira_backbone_validator (v0.1)

**Purpose:** Validate the immutable JIRA XML backbone and emit ML-ready variability features so that learning models focus on *content variability* rather than the *meta-structural layer*.

## Run
```bash
python -m jira_backbone_validator --in /path/to/JIRA.xml --out /path/to/outdir
```

## Outputs
- `backbone_report.csv` — per-issue backbone validation flags
- `backbone_summary.json` — dataset-level backbone stats
- `variability_features.parquet` — ML-ready features centered on variability (summary/description text + structural counts)
- `variability_schema.json` — schema dictionary for downstream pipelines

## Notes
- Supports both RSS (`rss/channel/item`) and `issues/issue` XML layouts.
- Date formats supported: RFC-822, ISO-8601 variants, classic JIRA textual.
- The variability extractor keeps clean text channels and a few compact structural signals.
