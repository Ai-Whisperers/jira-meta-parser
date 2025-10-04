"""Text cleaning and preprocessing utilities."""

import html
import re
from typing import Optional

from bs4 import BeautifulSoup


def clean_html(text: Optional[str], max_len: Optional[int] = None) -> str:
    """Clean HTML content from JIRA fields.

    Args:
        text: Raw HTML text
        max_len: Maximum length to truncate to

    Returns:
        Cleaned plain text
    """
    if not text:
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Parse and extract text (handles Confluence markup)
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Truncate if needed
    if max_len and len(cleaned) > max_len:
        cleaned = cleaned[:max_len]

    return cleaned


def extract_text_features(summary: Optional[str], description: Optional[str]) -> dict:
    """Extract text features for embedding and ranking.

    Args:
        summary: Ticket summary
        description: Ticket description

    Returns:
        Dictionary with cleaned text and length features
    """
    summary_clean = clean_html(summary)
    description_clean = clean_html(description)

    return {
        "summary_txt": summary_clean,
        "description_txt": description_clean,
        "summary_len": len(summary_clean),
        "description_len": len(description_clean),
    }


def normalize_categorical(value: Optional[str]) -> str:
    """Normalize categorical field values.

    Args:
        value: Raw categorical value

    Returns:
        Normalized value (lowercase, stripped)
    """
    if not value:
        return "unknown"

    return str(value).strip().lower()
