"""Tier 3 Dataset Extraction Script - Extract from ThePublicJiraDataset MongoDB.

This script extracts issues from the ThePublicJiraDataset MongoDB dump to create
a high-quality training corpus of 1-2.7M issues from 16 public Jira repositories.

PREREQUISITES:
1. MongoDB installed and running
2. Database restored: mongorestore --gzip --archive=datasets/merge/ThePublicJiraDataset/3.\ DataDump/mongodump-JiraReposAnon.archive
3. pymongo installed: pip install pymongo==3.11.3

USAGE:
    python scripts/extract_tier3_mongodb.py --strategy quality_filtered --limit 100000

STRATEGIES:
    - quality_filtered (RECOMMENDED): High-quality issues with complete fields
    - selective: Top 5 repos only (Apache, Mojang, Jira, MongoDB, Qt)
    - sample: 10% sample from each repo
    - full: All 2.7M issues
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm


class Tier3Extractor:
    """Extract Tier 3 training data from MongoDB."""

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        database: str = "JiraReposAnon",
        field_defs_path: str = "datasets/merge/ThePublicJiraDataset/0. DataDefinition/jira_field_information.json",
    ):
        """Initialize extractor."""
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Test connection
            self.db = self.client[database]
            print(f"✓ Connected to MongoDB: {database}")
        except Exception as e:
            print(f"✗ MongoDB connection failed: {e}")
            print(f"\nPlease ensure:")
            print(f"1. MongoDB is installed and running")
            print(f"2. Database restored with: mongorestore --gzip --archive=mongodump-JiraReposAnon.archive")
            raise

        # Load field definitions
        field_defs_file = Path(field_defs_path)
        if field_defs_file.exists():
            with open(field_defs_file, 'r') as f:
                self.field_defs = json.load(f)
            print(f"✓ Loaded field definitions from {field_defs_path}")
        else:
            print(f"⚠ Field definitions not found at {field_defs_path}")
            self.field_defs = {}

    def get_epic_field_id(self, repo: str) -> Optional[str]:
        """Find Epic Link custom field ID for a repository."""
        if repo not in self.field_defs:
            return None

        for field in self.field_defs[repo]:
            name_lower = field.get('name', '').lower()
            if 'epic' in name_lower and 'link' in name_lower:
                return field['id']

        return None

    def extract_issue(self, doc: Dict, repo: str) -> Optional[Dict]:
        """Extract issue data from MongoDB document."""
        try:
            fields = doc.get('fields', {})

            # Required fields
            key = doc.get('key')
            summary = fields.get('summary')

            if not key or not summary:
                return None

            # Issue type
            issuetype = fields.get('issuetype', {})
            issue_type = issuetype.get('name')

            # Status
            status = fields.get('status', {})
            status_name = status.get('name')

            # Priority (may be null)
            priority = fields.get('priority', {})
            priority_name = priority.get('name') if priority else None

            # Dates
            created = fields.get('created')
            updated = fields.get('updated')

            if not created or not updated:
                return None

            # Optional fields
            description = fields.get('description', '')

            assignee = fields.get('assignee', {})
            assignee_name = assignee.get('displayName') if assignee else None

            reporter = fields.get('reporter', {})
            reporter_name = reporter.get('displayName') if reporter else None

            # Components (join multiple)
            components = fields.get('components', [])
            components_str = ','.join([c.get('name', '') for c in components]) if components else ''

            # Labels
            labels = fields.get('labels', [])
            labels_str = ','.join(labels) if labels else ''

            # Parent (for subtasks)
            parent = fields.get('parent', {})
            parent_key = parent.get('key') if parent else None

            # Epic link (repo-specific custom field)
            epic_field_id = self.get_epic_field_id(repo)
            epic_link = fields.get(epic_field_id) if epic_field_id else None

            # Resolution
            resolution = fields.get('resolution', {})
            resolution_name = resolution.get('name') if resolution else None

            # Build result
            result = {
                'Issue key': key,
                'Summary': summary,
                'Issue Type': issue_type,
                'Status': status_name,
                'Priority': priority_name,
                'Created': created,
                'Updated': updated,
                'Description': description,
                'Assignee': assignee_name,
                'Reporter': reporter_name,
                'Components': components_str,
                'Labels': labels_str,
                'Parent key': parent_key,
                'Epic Link': epic_link,
                'Resolution': resolution_name,
                'Repository': repo,
            }

            return result

        except Exception as e:
            return None

    def build_quality_filter(self) -> Dict:
        """Build MongoDB query for quality-filtered extraction."""
        return {
            'fields.summary': {'$exists': True, '$ne': None, '$ne': ''},
            'fields.description': {'$exists': True, '$ne': None, '$ne': ''},
            'fields.priority.name': {'$exists': True, '$ne': None},
            'fields.created': {'$gte': '2015-01-01T00:00:00.000+0000'},
            'fields.issuetype.name': {
                '$in': ['Bug', 'Story', 'Task', 'Improvement', 'Epic', 'New Feature', 'Enhancement']
            }
        }

    def export_repository(
        self,
        repo: str,
        output_path: Path,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> int:
        """Export a single repository to CSV."""
        collection = self.db[repo]

        # Build query
        query = filters or {}

        # Get total count
        total = collection.count_documents(query)
        print(f"\n{repo}: {total:,} issues matching filters")

        if limit:
            total = min(total, limit)
            print(f"  Limiting to {limit:,} issues")

        if total == 0:
            return 0

        # Extract issues
        issues = []
        cursor = collection.find(query).limit(limit) if limit else collection.find(query)

        for doc in tqdm(cursor, total=total, desc=f"Extracting {repo}"):
            issue = self.extract_issue(doc, repo)
            if issue:
                issues.append(issue)

        # Write to CSV
        if issues:
            df = pd.DataFrame(issues)
            df.to_csv(output_path, index=False)
            print(f"  ✓ Exported {len(issues):,} issues to {output_path}")
            return len(issues)
        else:
            print(f"  ⚠ No issues to export for {repo}")
            return 0

    def export_all_repositories(
        self,
        output_dir: str = "datasets/prepared",
        strategy: str = "quality_filtered",
        merge_output: bool = True,
        limit_per_repo: Optional[int] = None,
    ) -> None:
        """Export all repositories using specified strategy."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("="*60)
        print("TIER 3 MONGODB EXTRACTION")
        print("="*60)
        print(f"Strategy: {strategy}")
        print(f"Output directory: {output_path}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Define filters based on strategy
        if strategy == "quality_filtered":
            filters = self.build_quality_filter()
            limit = limit_per_repo
        elif strategy == "sample":
            filters = {}
            limit = None  # Will calculate 10% per repo
        elif strategy == "selective":
            filters = {}
            limit = limit_per_repo
        else:  # full
            filters = {}
            limit = limit_per_repo

        # Get repositories
        repos = self.db.list_collection_names()
        print(f"\nFound {len(repos)} repositories in database")

        if strategy == "selective":
            # Only top repos
            repos = [r for r in ['Apache', 'Jira', 'MongoDB', 'Qt', 'Mojang'] if r in repos]
            print(f"Limiting to selective repos: {repos}")

        # Export each repository
        total_exported = 0
        csv_files = []
        repo_stats = {}

        for repo in repos:
            output_file = output_path / f"TIER3_{repo}.csv"

            # Calculate limit for sampling
            if strategy == "sample" and limit is None:
                total_docs = self.db[repo].count_documents({})
                repo_limit = int(total_docs * 0.1)
            else:
                repo_limit = limit

            count = self.export_repository(repo, output_file, filters, repo_limit)
            total_exported += count
            repo_stats[repo] = count

            if count > 0:
                csv_files.append(output_file)

        print(f"\n{'='*60}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total repositories: {len(repos)}")
        print(f"Total issues exported: {total_exported:,}")
        print(f"\nPer-repository breakdown:")
        for repo, count in sorted(repo_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {repo}: {count:,}")

        # Merge if requested
        if merge_output and csv_files:
            print(f"\nMerging all repositories into single CSV...")
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                dfs.append(df)

            merged_df = pd.concat(dfs, ignore_index=True)
            merged_path = output_path / "TIER3_MERGED.csv"
            merged_df.to_csv(merged_path, index=False)

            print(f"  ✓ Merged dataset: {len(merged_df):,} issues")
            print(f"  ✓ Saved to: {merged_path}")

            # Generate statistics
            print(f"\n{'='*60}")
            print("DATASET STATISTICS")
            print("="*60)
            print(f"Total issues: {len(merged_df):,}")
            print(f"\nBy repository:")
            print(merged_df['Repository'].value_counts())
            print(f"\nBy issue type:")
            print(merged_df['Issue Type'].value_counts().head(10))
            print(f"\nBy status:")
            print(merged_df['Status'].value_counts().head(10))
            print(f"\nField coverage:")
            for field in ['Summary', 'Description', 'Priority', 'Assignee']:
                coverage = (merged_df[field].notna().sum() / len(merged_df)) * 100
                print(f"  {field}: {coverage:.1f}%")

            # Save metadata
            metadata = {
                'tier': 'Tier 3',
                'description': 'Extracted from ThePublicJiraDataset MongoDB',
                'created': datetime.now().isoformat(),
                'strategy': strategy,
                'total_issues': len(merged_df),
                'total_repositories': len(repos),
                'repositories': repo_stats,
                'field_coverage': {
                    field: f"{(merged_df[field].notna().sum() / len(merged_df) * 100):.1f}%"
                    for field in ['Summary', 'Description', 'Priority', 'Assignee', 'Reporter']
                    if field in merged_df.columns
                },
                'distributions': {
                    'repository': merged_df['Repository'].value_counts().head(10).to_dict(),
                    'priority': merged_df['Priority'].value_counts().head(10).to_dict() if 'Priority' in merged_df.columns else {},
                    'status': merged_df['Status'].value_counts().head(10).to_dict() if 'Status' in merged_df.columns else {},
                    'issue_type': merged_df['Issue Type'].value_counts().head(10).to_dict() if 'Issue Type' in merged_df.columns else {}
                }
            }

            metadata_path = output_path / 'TIER3_MERGED_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"\n✓ Metadata saved: {metadata_path}")

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    """Main extraction function."""
    parser = argparse.ArgumentParser(description='Extract Tier 3 training data from MongoDB')
    parser.add_argument(
        '--strategy',
        choices=['quality_filtered', 'selective', 'sample', 'full'],
        default='quality_filtered',
        help='Extraction strategy (default: quality_filtered)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit issues per repository (optional)'
    )
    parser.add_argument(
        '--output',
        default='datasets/prepared',
        help='Output directory (default: datasets/prepared)'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Skip merging individual repo files'
    )

    args = parser.parse_args()

    try:
        extractor = Tier3Extractor()
        extractor.export_all_repositories(
            output_dir=args.output,
            strategy=args.strategy,
            merge_output=not args.no_merge,
            limit_per_repo=args.limit
        )
        extractor.close()

        print("\n✓ Tier 3 extraction completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Validate: python -m src.cli.dev validate datasets/prepared/TIER3_MERGED.csv")
        print(f"2. Update config: Set faiss.nlist to 16384, use_gpu to true")
        print(f"3. Train: python -m src.cli.dev full datasets/prepared/TIER3_MERGED.csv --skip-training=False")

    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check MongoDB is running: mongod --version")
        print(f"2. Check database restored: mongo JiraReposAnon --eval 'db.stats()'")
        print(f"3. Check pymongo installed: pip install pymongo==3.11.3")


if __name__ == "__main__":
    main()
