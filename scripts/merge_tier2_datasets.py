"""Tier 2 Dataset Merge Script - Merge Reference (602) + GFG (49K) datasets.

This script merges the reference dataset (datasets/JIRA.csv) with the GFG dataset
(datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv) to create a 49,602 issue training corpus.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def analyze_dataset(df, name):
    """Analyze dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    # Required fields
    required = ['Issue key', 'Summary', 'Issue Type', 'Status', 'Priority', 'Created', 'Updated']
    print(f"\nRequired field coverage:")
    for field in required:
        if field in df.columns:
            coverage = (df[field].notna().sum() / len(df)) * 100
            print(f"  {field}: {coverage:.1f}%")
        else:
            print(f"  {field}: MISSING")

    # Optional fields
    optional = ['Description', 'Assignee', 'Reporter', 'Components', 'Labels']
    print(f"\nOptional field coverage:")
    for field in optional:
        if field in df.columns:
            coverage = (df[field].notna().sum() / len(df)) * 100
            print(f"  {field}: {coverage:.1f}%")

    # Distributions
    if 'Priority' in df.columns:
        print(f"\nPriority distribution:")
        print(df['Priority'].value_counts().head(5))

    if 'Status' in df.columns:
        print(f"\nStatus distribution:")
        print(df['Status'].value_counts().head(10))

    if 'Issue Type' in df.columns:
        print(f"\nIssue Type distribution:")
        print(df['Issue Type'].value_counts().head(10))


def merge_datasets():
    """Merge reference and GFG datasets."""

    print("="*60)
    print("TIER 2 DATASET MERGE - Reference + GFG")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load datasets
    print("\n[1/6] Loading reference dataset...")
    ref = pd.read_csv('datasets/JIRA.csv')
    print(f"  Loaded: {len(ref):,} issues, {len(ref.columns)} columns")

    print("\n[2/6] Loading GFG dataset...")
    gfg = pd.read_csv('datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv')
    print(f"  Loaded: {len(gfg):,} issues, {len(gfg.columns)} columns")

    # Analyze before merge
    analyze_dataset(ref, "Reference (JIRA.csv)")
    analyze_dataset(gfg, "GFG_FINAL.csv")

    # Find common columns
    print("\n[3/6] Finding common columns...")
    common_cols = sorted(list(set(ref.columns) & set(gfg.columns)))
    print(f"  Common columns: {len(common_cols)}")

    # Check required fields
    required = ['Issue key', 'Summary', 'Issue Type', 'Status', 'Priority', 'Created', 'Updated']
    missing_required = [f for f in required if f not in common_cols]
    if missing_required:
        print(f"  ERROR: Missing required fields: {missing_required}")
        return None
    else:
        print(f"  ✓ All required fields present in common columns")

    # Handle missing priorities in GFG
    print("\n[4/6] Handling missing data...")
    gfg_missing_priority = gfg['Priority'].isna().sum()
    if gfg_missing_priority > 0:
        print(f"  GFG missing priorities: {gfg_missing_priority:,} ({gfg_missing_priority/len(gfg)*100:.1f}%)")
        print(f"  Imputing missing priorities to 'Medium'...")
        gfg['Priority'] = gfg['Priority'].fillna('Medium')
    else:
        print(f"  No missing priorities in GFG")

    # Check for duplicates
    print("\n[5/6] Checking for duplicates...")
    ref_keys = set(ref['Issue key'])
    gfg_keys = set(gfg['Issue key'])
    overlap = ref_keys & gfg_keys
    print(f"  Reference unique keys: {len(ref_keys):,}")
    print(f"  GFG unique keys: {len(gfg_keys):,}")
    print(f"  Overlapping keys: {len(overlap)}")

    if overlap:
        print(f"  Sample overlaps: {list(overlap)[:5]}")
        print(f"  Strategy: Keep reference version for duplicates")

    # Merge datasets
    print("\n[6/6] Merging datasets...")
    merged = pd.concat([ref[common_cols], gfg[common_cols]], ignore_index=True)
    print(f"  After concat: {len(merged):,} issues")

    # Remove duplicates (keep first = reference)
    merged = merged.drop_duplicates(subset='Issue key', keep='first')
    print(f"  After deduplication: {len(merged):,} issues")

    # Analyze merged dataset
    analyze_dataset(merged, "Merged Dataset")

    # Create output directory
    output_dir = Path('datasets/prepared')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged dataset
    output_path = output_dir / 'TIER2_MERGED.csv'
    print(f"\n[SAVING] Writing to {output_path}...")
    merged.to_csv(output_path, index=False)
    print(f"  Saved: {len(merged):,} issues, {len(merged.columns)} columns")

    # Create metadata
    metadata = {
        'tier': 'Tier 2',
        'description': 'Merged Reference (JIRA.csv) + GFG (GFG_FINAL.csv) datasets',
        'created': datetime.now().isoformat(),
        'source_datasets': {
            'reference': {
                'path': 'datasets/JIRA.csv',
                'issues': len(ref),
                'columns': len(ref.columns)
            },
            'gfg': {
                'path': 'datasets/merge/CSV_JIRA_ARCHIVE/GFG_FINAL.csv',
                'issues': len(gfg),
                'columns': len(gfg.columns)
            }
        },
        'merged_dataset': {
            'path': str(output_path),
            'total_issues': len(merged),
            'unique_issues': len(merged),
            'columns': len(merged.columns),
            'common_columns_used': len(common_cols),
            'duplicates_removed': len(ref) + len(gfg) - len(merged)
        },
        'field_coverage': {
            field: f"{(merged[field].notna().sum() / len(merged) * 100):.1f}%"
            for field in required if field in merged.columns
        },
        'distributions': {
            'priority': merged['Priority'].value_counts().head(10).to_dict() if 'Priority' in merged.columns else {},
            'status': merged['Status'].value_counts().head(10).to_dict() if 'Status' in merged.columns else {},
            'issue_type': merged['Issue Type'].value_counts().head(10).to_dict() if 'Issue Type' in merged.columns else {}
        }
    }

    metadata_path = output_dir / 'TIER2_MERGED_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    # Create summary
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    print(f"Reference dataset: {len(ref):,} issues")
    print(f"GFG dataset: {len(gfg):,} issues")
    print(f"Merged dataset: {len(merged):,} issues")
    print(f"Increase from baseline: {len(merged) / 602:.1f}x")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    return merged, metadata


if __name__ == "__main__":
    merged_df, metadata = merge_datasets()

    if merged_df is not None:
        print("\n✓ Tier 2 dataset merge completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Validate: python -m src.cli.dev validate datasets/prepared/TIER2_MERGED.csv")
        print(f"2. Update config: Set faiss.nlist to 8192 in config/default.yaml")
        print(f"3. Train: python -m src.cli.dev full datasets/prepared/TIER2_MERGED.csv --skip-training=False")
    else:
        print("\n✗ Merge failed. Check errors above.")
