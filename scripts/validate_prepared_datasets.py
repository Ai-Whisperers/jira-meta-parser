"""Validate prepared training datasets for quality and completeness.

This script validates the prepared Tier 2 and Tier 3 datasets to ensure they meet
the requirements for training the JIRA Ticket Meta Parser.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def validate_dataset(csv_path: str, tier_name: str) -> dict:
    """Validate a dataset and return quality report."""

    print(f"\n{'='*60}")
    print(f"VALIDATING: {tier_name}")
    print(f"{'='*60}")
    print(f"File: {csv_path}")

    if not Path(csv_path).exists():
        print(f"✗ File not found: {csv_path}")
        return {'status': 'NOT_FOUND', 'path': csv_path}

    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded: {len(df):,} issues, {len(df.columns)} columns")

    # Check required fields
    required_fields = ['Issue key', 'Summary', 'Issue Type', 'Status', 'Priority', 'Created', 'Updated']
    optional_fields = ['Description', 'Assignee', 'Reporter', 'Components', 'Labels']

    # Required field validation
    print(f"\n[1/5] Required Fields (7 fields):")
    required_ok = True
    required_coverage = {}

    for field in required_fields:
        if field not in df.columns:
            print(f"  ✗ {field}: MISSING COLUMN")
            required_ok = False
        else:
            coverage = (df[field].notna().sum() / len(df)) * 100
            required_coverage[field] = coverage
            status = "✓" if coverage == 100.0 else "⚠"
            print(f"  {status} {field}: {coverage:.1f}%")

            if coverage < 100.0:
                print(f"     Warning: {(100-coverage):.1f}% missing values")

    # Optional field validation
    print(f"\n[2/5] Optional Fields (5 fields):")
    optional_coverage = {}

    for field in optional_fields:
        if field not in df.columns:
            print(f"  - {field}: Not present")
        else:
            coverage = (df[field].notna().sum() / len(df)) * 100
            optional_coverage[field] = coverage
            status = "✓" if coverage > 60 else "⚠" if coverage > 30 else "✗"
            print(f"  {status} {field}: {coverage:.1f}%")

    # Data distributions
    print(f"\n[3/5] Data Distributions:")

    if 'Priority' in df.columns:
        priority_counts = df['Priority'].value_counts()
        print(f"  Priority (top 5):")
        for priority, count in priority_counts.head(5).items():
            pct = (count / len(df)) * 100
            print(f"    {priority}: {count:,} ({pct:.1f}%)")

    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts()
        print(f"  Status (top 10):")
        for status, count in status_counts.head(10).items():
            pct = (count / len(df)) * 100
            print(f"    {status}: {count:,} ({pct:.1f}%)")

    if 'Issue Type' in df.columns:
        type_counts = df['Issue Type'].value_counts()
        print(f"  Issue Type (top 5):")
        for itype, count in type_counts.head(5).items():
            pct = (count / len(df)) * 100
            print(f"    {itype}: {count:,} ({pct:.1f}%)")

    # Data quality checks
    print(f"\n[4/5] Data Quality Checks:")

    # Check for duplicates
    if 'Issue key' in df.columns:
        duplicates = len(df) - df['Issue key'].nunique()
        if duplicates > 0:
            print(f"  ⚠ Duplicate keys: {duplicates:,} ({duplicates/len(df)*100:.1f}%)")
        else:
            print(f"  ✓ No duplicate keys")

    # Check for empty summaries
    if 'Summary' in df.columns:
        empty_summaries = (df['Summary'].isna() | (df['Summary'] == '')).sum()
        if empty_summaries > 0:
            print(f"  ⚠ Empty summaries: {empty_summaries:,} ({empty_summaries/len(df)*100:.1f}%)")
        else:
            print(f"  ✓ All issues have summaries")

    # Check description quality
    if 'Description' in df.columns:
        desc_present = df['Description'].notna().sum()
        desc_pct = (desc_present / len(df)) * 100
        if desc_pct < 60:
            print(f"  ⚠ Low description coverage: {desc_pct:.1f}%")
        else:
            print(f"  ✓ Good description coverage: {desc_pct:.1f}%")

    # Calculate overall quality score
    print(f"\n[5/5] Overall Quality Assessment:")

    quality_score = 0

    # Required fields (40 points)
    if all(required_coverage.get(f, 0) == 100.0 for f in required_fields):
        quality_score += 40
        print(f"  ✓ Required fields: 40/40")
    else:
        avg_coverage = sum(required_coverage.values()) / len(required_coverage)
        points = int((avg_coverage / 100) * 40)
        quality_score += points
        print(f"  ⚠ Required fields: {points}/40")

    # Description coverage (20 points)
    desc_coverage = optional_coverage.get('Description', 0)
    if desc_coverage >= 80:
        quality_score += 20
        print(f"  ✓ Description: 20/20")
    elif desc_coverage >= 60:
        quality_score += 15
        print(f"  ⚠ Description: 15/20")
    else:
        points = int((desc_coverage / 100) * 20)
        quality_score += points
        print(f"  ⚠ Description: {points}/20")

    # Priority coverage (20 points)
    priority_coverage = required_coverage.get('Priority', 0)
    if priority_coverage >= 95:
        quality_score += 20
        print(f"  ✓ Priority: 20/20")
    else:
        points = int((priority_coverage / 100) * 20)
        quality_score += points
        print(f"  ⚠ Priority: {points}/20")

    # No duplicates (10 points)
    if duplicates == 0:
        quality_score += 10
        print(f"  ✓ No duplicates: 10/10")
    else:
        print(f"  ✗ Duplicates found: 0/10")

    # Size appropriateness (10 points)
    if len(df) >= 500:
        quality_score += 10
        print(f"  ✓ Dataset size: 10/10")
    elif len(df) >= 100:
        quality_score += 5
        print(f"  ⚠ Dataset size: 5/10")
    else:
        print(f"  ✗ Dataset size: 0/10")

    # Final assessment
    print(f"\n{'='*60}")
    print(f"QUALITY SCORE: {quality_score}/100")

    if quality_score >= 90:
        status = "EXCELLENT"
        recommendation = "✓ READY FOR TRAINING"
    elif quality_score >= 75:
        status = "GOOD"
        recommendation = "✓ READY FOR TRAINING"
    elif quality_score >= 60:
        status = "ACCEPTABLE"
        recommendation = "⚠ CONSIDER IMPROVEMENTS"
    else:
        status = "POOR"
        recommendation = "✗ NEEDS IMPROVEMENT"

    print(f"STATUS: {status}")
    print(f"RECOMMENDATION: {recommendation}")
    print("="*60)

    # Return validation report
    return {
        'tier': tier_name,
        'path': csv_path,
        'status': status,
        'quality_score': quality_score,
        'total_issues': len(df),
        'total_columns': len(df.columns),
        'required_coverage': required_coverage,
        'optional_coverage': optional_coverage,
        'duplicates': duplicates if 'Issue key' in df.columns else None,
        'recommendation': recommendation,
        'validated_at': datetime.now().isoformat()
    }


def main():
    """Main validation function."""

    print("="*60)
    print("DATASET VALIDATION SUITE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    datasets = {
        'Tier 1 (Baseline)': 'datasets/JIRA.csv',
        'Tier 2 (Merged)': 'datasets/prepared/TIER2_MERGED.csv',
        'Tier 3 (MongoDB)': 'datasets/prepared/TIER3_MERGED.csv'
    }

    reports = {}

    for tier_name, csv_path in datasets.items():
        report = validate_dataset(csv_path, tier_name)
        reports[tier_name] = report

    # Save combined validation report
    output_dir = Path('datasets/prepared')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(reports, f, indent=2)

    print(f"\n✓ Validation reports saved to: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print("="*60)

    for tier_name, report in reports.items():
        if report['status'] != 'NOT_FOUND':
            print(f"{tier_name}:")
            print(f"  Issues: {report['total_issues']:,}")
            print(f"  Quality: {report['quality_score']}/100 ({report['status']})")
            print(f"  {report['recommendation']}")
        else:
            print(f"{tier_name}: File not found")

    print("="*60)


if __name__ == "__main__":
    main()
