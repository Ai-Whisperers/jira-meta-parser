"""CSV adapter - chunked parser for JIRA CSV exports."""

from typing import Dict, Iterator, List, Optional

import pandas as pd


class CSVAdapter:
    """Chunked CSV parser for JIRA exports."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """Initialize CSV adapter.

        Args:
            column_mapping: Custom column name mapping (default uses JIRA standard)
        """
        # Default JIRA CSV column mapping
        self.column_mapping = column_mapping or {
            "Issue key": "key",
            "Summary": "summary",
            "Issue Type": "type",
            "Status": "status",
            "Priority": "priority",
            "Created": "created",
            "Updated": "updated",
            "Assignee": "assignee",
            "Reporter": "reporter",
            "Components": "components",
            "Labels": "labels",
            "Parent": "parent",
            "Epic Link": "epic_link",
            "Sprint": "sprint",
            "Rank": "rank",
            "Description": "description",
        }

    def parse(self, filepath: str, chunksize: int = 1000) -> Iterator[Dict[str, any]]:
        """Parse CSV file and yield canonical issue dictionaries.

        Args:
            filepath: Path to JIRA CSV export
            chunksize: Number of rows to read per chunk (memory efficiency)

        Yields:
            Canonical issue dictionaries
        """
        # Read CSV in chunks to handle large files
        for chunk in pd.read_csv(filepath, chunksize=chunksize, encoding="utf-8"):
            # Rename columns to canonical names
            chunk = self._normalize_columns(chunk)

            # Convert each row to canonical dict
            for _, row in chunk.iterrows():
                yield self._row_to_canonical(row)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to canonical names.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with normalized column names
        """
        # Find matching columns (case-insensitive)
        rename_map = {}
        for col in df.columns:
            for csv_col, canonical in self.column_mapping.items():
                if col.strip().lower() == csv_col.lower():
                    rename_map[col] = canonical
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _row_to_canonical(self, row: pd.Series) -> Dict[str, any]:
        """Convert DataFrame row to canonical issue dictionary.

        Args:
            row: DataFrame row

        Returns:
            Canonical issue dictionary
        """
        issue = {}

        # Basic fields (scalar)
        scalar_fields = [
            "key",
            "summary",
            "type",
            "status",
            "priority",
            "created",
            "updated",
            "description",
            "assignee",
            "reporter",
            "parent",
            "epic_link",
            "sprint",
            "rank",
        ]

        for field in scalar_fields:
            value = row.get(field)
            # Convert NaN to None
            issue[field] = None if pd.isna(value) else str(value).strip()

        # Multi-valued fields (split on semicolon)
        issue["labels"] = self._split_multi(row.get("labels"))
        issue["components"] = self._split_multi(row.get("components"))

        # Custom fields count (if CSV has customfield columns)
        customfield_cols = [c for c in row.index if c.startswith("customfield")]
        issue["customfield_count"] = len(customfield_cols)

        # Issue links (CSV typically doesn't have structured links, so empty)
        # If links are present in specific columns, handle them here
        issue["issuelinks"] = []

        return issue

    @staticmethod
    def _split_multi(value: any) -> List[str]:
        """Split multi-valued field (semicolon or comma separated).

        Args:
            value: Raw field value

        Returns:
            List of individual values
        """
        if pd.isna(value) or not value:
            return []

        # Split on semicolon (JIRA export standard) or comma
        separator = ";" if ";" in str(value) else ","
        values = [v.strip() for v in str(value).split(separator) if v.strip()]

        return values
