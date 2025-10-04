"""XML adapter - streaming parser for JIRA XML exports (RSS/issue format)."""

from typing import Dict, Iterator, List, Optional
from xml.etree import ElementTree as ET


class XMLAdapter:
    """Streaming XML parser for JIRA exports."""

    def __init__(self):
        """Initialize XML adapter."""
        # XPath-like mappings for common JIRA XML structures
        self.field_map = {
            "key": ["key", "issue/key"],
            "summary": ["summary", "title", "issue/summary"],
            "type": ["type", "issue/type"],
            "status": ["status", "issue/status"],
            "priority": ["priority", "issue/priority"],
            "created": ["created", "pubDate", "issue/created"],
            "updated": ["updated", "issue/updated"],
            "description": ["description", "issue/description"],
            "assignee": ["assignee", "issue/assignee"],
            "reporter": ["reporter", "issue/reporter"],
            "parent": ["parent", "issue/parent"],
        }

    def parse(self, filepath: str) -> Iterator[Dict[str, any]]:
        """Parse XML file and yield canonical issue dictionaries.

        Args:
            filepath: Path to JIRA XML export

        Yields:
            Canonical issue dictionaries
        """
        # Detect format (RSS vs. direct issues)
        context = ET.iterparse(filepath, events=("start", "end"))
        context = iter(context)

        _, root = next(context)

        for event, elem in context:
            if event == "end":
                # RSS format: <item> elements
                if elem.tag == "item":
                    yield self._parse_item(elem)
                    elem.clear()

                # Direct issue format: <issue> elements
                elif elem.tag == "issue":
                    yield self._parse_item(elem)
                    elem.clear()

                # Clear root periodically to free memory
                if elem == root:
                    root.clear()

    def _parse_item(self, elem: ET.Element) -> Dict[str, any]:
        """Parse a single item/issue element to canonical dict.

        Args:
            elem: XML element (item or issue)

        Returns:
            Canonical issue dictionary
        """
        issue = {}

        # Extract basic fields
        for field, xpaths in self.field_map.items():
            issue[field] = self._find_text(elem, xpaths)

        # Extract multi-valued fields
        issue["labels"] = self._extract_multi(elem, ["labels/label", "label"])
        issue["components"] = self._extract_multi(elem, ["components/component", "component"])

        # Extract custom fields (count only for variability)
        issue["customfield_count"] = len(elem.findall(".//customfield"))

        # Extract issue links
        issue["issuelinks"] = self._extract_links(elem)

        return issue

    def _find_text(self, elem: ET.Element, xpaths: List[str]) -> Optional[str]:
        """Find first matching text from list of XPaths.

        Args:
            elem: XML element to search
            xpaths: List of XPath expressions to try

        Returns:
            Text content or None
        """
        for xpath in xpaths:
            found = elem.find(xpath)
            if found is not None and found.text:
                return found.text.strip()

        return None

    def _extract_multi(self, elem: ET.Element, xpaths: List[str]) -> List[str]:
        """Extract multiple values from repeating elements.

        Args:
            elem: XML element to search
            xpaths: List of XPath expressions to try

        Returns:
            List of text values
        """
        values = []
        for xpath in xpaths:
            for found in elem.findall(xpath):
                if found.text:
                    values.append(found.text.strip())

        return values

    def _extract_links(self, elem: ET.Element) -> List[Dict[str, str]]:
        """Extract issue links.

        Args:
            elem: XML element to search

        Returns:
            List of link dictionaries with 'type' and 'key'
        """
        links = []

        # Try different link structures
        for link in elem.findall(".//issuelinktype"):
            # Inward links
            for inward in link.findall(".//inwardlinks/issuelink/issuekey"):
                if inward.text:
                    links.append(
                        {
                            "type": link.findtext("name", "unknown"),
                            "direction": "inward",
                            "key": inward.text.strip(),
                        }
                    )

            # Outward links
            for outward in link.findall(".//outwardlinks/issuelink/issuekey"):
                if outward.text:
                    links.append(
                        {
                            "type": link.findtext("name", "unknown"),
                            "direction": "outward",
                            "key": outward.text.strip(),
                        }
                    )

        # Alternative: direct issuelinks (some exports)
        for link in elem.findall(".//issuelink"):
            key_elem = link.find("issuekey")
            if key_elem is not None and key_elem.text:
                links.append({"type": "link", "direction": "unknown", "key": key_elem.text.strip()})

        return links
