
import argparse
from pathlib import Path
from .validator import validate_backbone
from .features import extract_variability_features

def main():
    ap = argparse.ArgumentParser(description="JIRA XML backbone validator + variability feature extractor")
    ap.add_argument("--in", dest="xml_in", required=True, help="Path to JIRA XML export")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    xml_in = Path(args.xml_in)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    back_csv = out_dir / "backbone_report.csv"
    back_json = out_dir / "backbone_summary.json"
    summary, rows = validate_backbone(xml_in, back_csv, back_json)

    feat_parquet = out_dir / "variability_features.parquet"
    schema_json = out_dir / "variability_schema.json"
    df, schema = extract_variability_features(xml_in, feat_parquet, schema_json)

    print("Done.")
    print(f"- Backbone report: {back_csv}")
    print(f"- Backbone summary: {back_json}")
    print(f"- Variability features: {feat_parquet}")
    print(f"- Variability schema: {schema_json}")

if __name__ == "__main__":
    main()
