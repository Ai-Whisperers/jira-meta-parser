"""Backbone validator - dual-format (XML+CSV) with O(n) streaming validation."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser

from ..adapters import CSVAdapter, XMLAdapter


class BackboneValidator:
    """Validates JIRA exports against backbone schema (format-agnostic)."""

    def __init__(
        self,
        key_regex: str,
        date_formats: List[str],
        required_fields: List[str],
        optional_fields: List[str],
        link_policy: str = "strict",
        csv_column_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize validator.

        Args:
            key_regex: Regex pattern for issue keys
            date_formats: List of accepted date format strings
            required_fields: Fields that must be present
            optional_fields: Recognized optional fields
            link_policy: 'strict' or 'allow_external'
            csv_column_mapping: Custom CSV column mapping
        """
        self.key_regex = re.compile(key_regex)
        self.date_formats = date_formats
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.link_policy = link_policy

        # Initialize adapters
        self.xml_adapter = XMLAdapter()
        self.csv_adapter = CSVAdapter(column_mapping=csv_column_mapping)

    def validate_file(
        self, filepath: str, fmt: str = "auto"
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Validate JIRA export file.

        Args:
            filepath: Path to JIRA export (XML or CSV)
            fmt: Format hint ('auto', 'xml', 'csv')

        Returns:
            Tuple of (report_df, summary_dict)
        """
        # Detect format
        if fmt == "auto":
            fmt = "xml" if filepath.endswith(".xml") else "csv"

        # Load rows
        rows = self._load_rows(filepath, fmt)

        # Validate (two-pass for link integrity)
        report, summary = self._validate_rows(rows)

        return report, summary

    def _load_rows(self, filepath: str, fmt: str) -> List[Dict[str, any]]:
        """Load and cache all rows from file.

        Args:
            filepath: Path to file
            fmt: Format ('xml' or 'csv')

        Returns:
            List of canonical issue dictionaries
        """
        if fmt == "xml":
            return list(self.xml_adapter.parse(filepath))
        elif fmt == "csv":
            return list(self.csv_adapter.parse(filepath))
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _validate_rows(
        self, rows: List[Dict[str, any]]
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Validate rows with two-pass link checking.

        Args:
            rows: List of canonical issue dicts

        Returns:
            Tuple of (report DataFrame, summary dict)
        """
        # First pass: collect all keys
        key_set = set()
        key_counts = {}
        for row in rows:
            key = row.get("key")
            if key:
                key_set.add(key)
                key_counts[key] = key_counts.get(key, 0) + 1

        # Second pass: validate each row
        report_rows = []
        errors = {
            "missing_required": 0,
            "bad_key_format": 0,
            "duplicate_key": 0,
            "bad_dates": 0,
            "link_reference_errors": 0,
            "invalid_parent": 0,
        }

        seen_keys = set()

        for row in rows:
            flags = self._validate_row(row, key_set, key_counts, seen_keys)

            # Count errors
            if not flags["required_ok"]:
                errors["missing_required"] += 1
            if not flags["key_format_ok"]:
                errors["bad_key_format"] += 1
            if not flags["unique_key_ok"]:
                errors["duplicate_key"] += 1
            if not flags["dates_ok"]:
                errors["bad_dates"] += 1
            if flags["link_ref_errors"] > 0:
                errors["link_reference_errors"] += flags["link_ref_errors"]
            if not flags["parent_ok"]:
                errors["invalid_parent"] += 1

            # Build report row
            report_rows.append(
                {
                    "key": row.get("key"),
                    "required_ok": flags["required_ok"],
                    "key_format_ok": flags["key_format_ok"],
                    "unique_key_ok": flags["unique_key_ok"],
                    "created_ok": flags["created_ok"],
                    "updated_ok": flags["updated_ok"],
                    "dates_ok": flags["dates_ok"],
                    "link_ref_errors": flags["link_ref_errors"],
                    "parent_ok": flags["parent_ok"],
                    "summary": row.get("summary", "")[:100],  # Truncate for report
                    "type": row.get("type"),
                    "status": row.get("status"),
                    "priority": row.get("priority"),
                }
            )

        # Create report DataFrame
        report_df = pd.DataFrame(report_rows)

        # Create summary
        summary = {
            "issues_count": len(rows),
            "unique_keys_count": len(key_set),
            "errors": errors,
            "schema_backbone": {
                "required_fields": self.required_fields,
                "optional_fields": self.optional_fields,
            },
        }

        return report_df, summary

    def _validate_row(
        self,
        row: Dict[str, any],
        key_set: set,
        key_counts: Dict[str, int],
        seen_keys: set,
    ) -> Dict[str, any]:
        """Validate a single row.

        Args:
            row: Issue dictionary
            key_set: Set of all keys in dataset
            key_counts: Count of each key
            seen_keys: Keys seen so far (for duplicate detection)

        Returns:
            Dictionary of validation flags
        """
        flags = {}

        # Required fields check
        flags["required_ok"] = all(
            row.get(field) for field in self.required_fields
        )

        # Key format check
        key = row.get("key") or ""
        flags["key_format_ok"] = bool(self.key_regex.match(key))

        # Unique key check
        flags["unique_key_ok"] = (
            key not in seen_keys if key else False
        )
        if key:
            seen_keys.add(key)

        # Date checks
        flags["created_ok"] = self._parse_date_ok(row.get("created"))
        flags["updated_ok"] = self._parse_date_ok(row.get("updated"))
        flags["dates_ok"] = flags["created_ok"] and flags["updated_ok"]

        # Link reference checks
        link_errors = 0
        for link in row.get("issuelinks", []):
            link_key = link.get("key")
            if link_key and link_key not in key_set:
                if self.link_policy == "strict":
                    link_errors += 1

        flags["link_ref_errors"] = link_errors

        # Parent check
        parent_key = row.get("parent")
        flags["parent_ok"] = (
            not parent_key or parent_key in key_set
        )

        return flags

    def _parse_date_ok(self, date_str: Optional[str]) -> bool:
        """Check if date string is parseable.

        Args:
            date_str: Date string to parse

        Returns:
            True if parseable, False otherwise
        """
        if not date_str:
            return False

        # Try explicit formats first
        for fmt in self.date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except (ValueError, TypeError):
                continue

        # Fallback to dateutil parser (flexible)
        try:
            date_parser.parse(date_str)
            return True
        except (ValueError, TypeError, date_parser.ParserError):
            return False


def create_validator(config: Dict[str, any]) -> BackboneValidator:
    """Factory function to create validator from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured BackboneValidator instance
    """
    validator_cfg = config.get("validator", {})

    return BackboneValidator(
        key_regex=validator_cfg.get("key_regex", r"^[A-Z][A-Z0-9_]+-\d+$"),
        date_formats=validator_cfg.get("date_formats", []),
        required_fields=validator_cfg.get("required_fields", []),
        optional_fields=validator_cfg.get("optional_fields", []),
        link_policy=validator_cfg.get("link_policy", "strict"),
        csv_column_mapping=validator_cfg.get("csv_column_mapping"),
    )
