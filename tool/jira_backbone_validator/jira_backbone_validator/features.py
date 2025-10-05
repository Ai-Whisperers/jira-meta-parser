
import re, json, html
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd

TAG_RE = re.compile(r"<.*?>", flags=re.S)
WS_RE = re.compile(r"\s+")

def _strip_html(x: str) -> str:
    if not x: return ""
    x = html.unescape(x)
    x = TAG_RE.sub(" ", x)
    x = WS_RE.sub(" ", x).strip()
    return x

def parse_items(root):
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//issue")
    return items

def get_text(node, tag):
    el = node.find(tag)
    return el.text.strip() if (el is not None and el.text) else ""

def extract_variability_features(xml_path: Path, out_parquet: Path, out_schema: Path):
    """Extract ML-ready features centered on variability (content), not backbone."""
    xml_path = Path(xml_path)
    import pandas as pd
    from xml.etree import ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    items = parse_items(root)

    rows = []
    for it in items:
        key = get_text(it, "key")
        summary = get_text(it, "summary")
        desc = get_text(it, "description")
        itype = get_text(it, "type")
        status = get_text(it, "status")
        priority = get_text(it, "priority")

        summary_clean = _strip_html(summary)
        desc_clean = _strip_html(desc)

        labels = it.find("labels")
        label_count = len(labels.findall("label")) if labels is not None else 0

        components = it.find("components")
        comp_count = len(components.findall("component")) if components is not None else 0

        customfields = it.find("customfields")
        cf_count = len(customfields.findall("customfield")) if customfields is not None else 0

        issuelinks = it.find("issuelinks")
        link_count = len(issuelinks.findall("link")) if issuelinks is not None else 0

        rows.append({
            "key": key,
            "type": itype,
            "status": status,
            "priority": priority,
            "summary_txt": summary_clean,
            "description_txt": desc_clean,
            "label_count": label_count,
            "component_count": comp_count,
            "customfield_count": cf_count,
            "link_count": link_count
        })

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    schema = {
        "version": "0.1.0",
        "fields": [
            {"name":"key","type":"string","role":"id"},
            {"name":"type","type":"category","role":"anchor"},
            {"name":"status","type":"category","role":"anchor"},
            {"name":"priority","type":"category","role":"anchor"},
            {"name":"summary_txt","type":"text","role":"variability"},
            {"name":"description_txt","type":"text","role":"variability"},
            {"name":"label_count","type":"int","role":"variability"},
            {"name":"component_count","type":"int","role":"variability"},
            {"name":"customfield_count","type":"int","role":"variability"},
            {"name":"link_count","type":"int","role":"variability"}
        ],
        "notes": "Focus on content variability; backbone validation handled separately."
    }
    with open(out_schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return df, schema
