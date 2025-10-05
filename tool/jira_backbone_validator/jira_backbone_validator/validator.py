
import re, csv, json
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")

RFC822 = "%a, %d %b %Y %H:%M:%S %z"
ISO1   = "%Y-%m-%dT%H:%M:%S%z"
ISO2   = "%Y-%m-%dT%H:%M:%S.%f%z"
ISO3   = "%Y-%m-%d %H:%M:%S%z"
JIRA   = "%a %b %d %H:%M:%S %Z %Y"
YMD    = "%Y-%m-%d"

DATE_FORMATS = [RFC822, ISO1, ISO2, ISO3, JIRA, YMD]

def parse_items(root):
    """Support both RSS (rss/channel/item) and issues/issue layouts."""
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//issue")
    return items

def get_text(node, tag):
    el = node.find(tag)
    return el.text.strip() if (el is not None and el.text) else ""

def basic_date_ok(s):
    if not s:
        return False
    for f in DATE_FORMATS:
        try:
            datetime.strptime(s, f)
            return True
        except Exception:
            continue
    return False

def validate_backbone(xml_path: Path, out_csv: Path, out_json: Path):
    """Validate immutable JIRA XML backbone fields and write artifacts."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    items = parse_items(root)

    report_rows = []
    seen_keys = set()
    missing_required = 0
    bad_key_format = 0
    bad_dates = 0
    link_ref_errors = 0

    # Collect keys for link validation
    keys_all = set()
    for it in items:
        key = get_text(it, "key")
        if key:
            keys_all.add(key)

    for it in items:
        key = get_text(it, "key")
        summary = get_text(it, "summary")
        itype = get_text(it, "type")
        status = get_text(it, "status")
        priority = get_text(it, "priority")
        created = get_text(it, "created")
        updated = get_text(it, "updated")

        required_present = all([key, summary, itype, status, priority, created, updated])
        if not required_present:
            missing_required += 1

        key_ok = bool(KEY_RE.match(key)) if key else False
        if key and not key_ok:
            bad_key_format += 1

        unique_ok = key not in seen_keys if key else False
        if key:
            seen_keys.add(key)

        created_ok = basic_date_ok(created)
        updated_ok = basic_date_ok(updated)
        date_ok = created_ok and updated_ok
        if not date_ok:
            bad_dates += 1

        link_errors = 0
        issuelinks = it.find("issuelinks")
        if issuelinks is not None:
            for link in issuelinks.findall("link"):
                target = get_text(link, "key")
                if target and target not in keys_all:
                    link_errors += 1
        if link_errors:
            link_ref_errors += link_errors

        report_rows.append({
            "key": key,
            "summary": (summary or "")[:140],
            "type": itype,
            "status": status,
            "priority": priority,
            "created_ok": created_ok,
            "updated_ok": updated_ok,
            "dates_ok": date_ok,
            "required_ok": required_present,
            "key_format_ok": key_ok,
            "unique_key_ok": unique_ok,
            "link_ref_errors": link_errors
        })

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if report_rows:
            headers = list(report_rows[0].keys())
        else:
            headers = ["key","summary","type","status","priority","created_ok","updated_ok","dates_ok","required_ok","key_format_ok","unique_key_ok","link_ref_errors"]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in report_rows:
            w.writerow(row)

    summary = {
        "file": str(xml_path),
        "issues_count": len(items),
        "missing_required_count": missing_required,
        "bad_key_format_count": bad_key_format,
        "bad_dates_count": bad_dates,
        "link_reference_errors": link_ref_errors,
        "unique_keys_count": len(seen_keys),
        "schema_backbone": {
            "required_fields": ["key","summary","type","status","priority","created","updated"],
            "optional_fields": ["description","reporter","assignee","labels","components","customfields","issuelinks"]
        }
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, report_rows
