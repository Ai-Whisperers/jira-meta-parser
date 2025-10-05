#!/usr/bin/env python
"""Verify installation and environment setup."""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.9."""
    print("Checking Python version...")
    version = sys.version_info
    if version >= (3, 9):
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need >= 3.9)")
        return False


def check_dependencies():
    """Check required packages are installed."""
    print("\nChecking dependencies...")
    required = [
        "numpy",
        "pandas",
        "pyarrow",
        "sentence_transformers",
        "faiss",
        "lightgbm",
        "lxml",
        "yaml",
        "click",
    ]

    all_ok = True
    for package in required:
        try:
            if package == "faiss":
                # FAISS can be faiss-cpu or faiss-gpu
                try:
                    import faiss
                except ImportError:
                    import faiss_cpu as faiss
            else:
                __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
            all_ok = False

    return all_ok


def check_directories():
    """Check required directories exist."""
    print("\nChecking project structure...")
    required_dirs = [
        "config",
        "src/core",
        "src/adapters",
        "src/utils",
        "src/cli",
        "./models/model-files/all-MiniLM-L6-v2",
        "./datasets",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_ok = False

    return all_ok


def check_config():
    """Check config file exists and loads."""
    print("\nChecking configuration...")
    config_path = Path("config/default.yaml")

    if not config_path.exists():
        print(f"  ✗ config/default.yaml (missing)")
        return False

    try:
        from src.utils import Config

        config = Config()
        print(f"  ✓ config/default.yaml loads successfully")

        # Check key sections
        required_sections = ["validator", "features", "embeddings", "faiss", "ranker"]
        for section in required_sections:
            if section in config.to_dict():
                print(f"    ✓ {section} section present")
            else:
                print(f"    ✗ {section} section missing")
                return False

        return True

    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_imports():
    """Check all core modules import successfully."""
    print("\nChecking imports...")

    try:
        from src.pipeline import JIRAPipeline
        from src.utils import Config

        config = Config()
        pipeline = JIRAPipeline(config.to_dict())
        print("  ✓ All core modules import successfully")
        print("  ✓ Pipeline initializes successfully")
        return True

    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def check_sample_data():
    """Check if sample data exists."""
    print("\nChecking sample data...")
    xml_path = Path("datasets/JIRA.xml")
    csv_path = Path("datasets/csv-exported-from-xml/JIRA.csv")

    has_xml = xml_path.exists()
    has_csv = csv_path.exists()

    if has_xml:
        size_mb = xml_path.stat().st_size / 1024 / 1024
        print(f"  ✓ JIRA.xml ({size_mb:.1f} MB)")
    else:
        print(f"  ℹ JIRA.xml (not present, will need your own data)")

    if has_csv:
        size_mb = csv_path.stat().st_size / 1024 / 1024
        print(f"  ✓ JIRA.csv ({size_mb:.1f} MB)")
    else:
        print(f"  ℹ JIRA.csv (not present, will need your own data)")

    return has_xml or has_csv


def main():
    """Run all checks."""
    print("=" * 60)
    print("JIRA Ticket Meta Parser - Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_directories),
        ("Configuration", check_config),
        ("Module Imports", check_imports),
        ("Sample Data", check_sample_data),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🚀 All checks passed! Ready to run pipeline.")
        print("\nQuick start:")
        print("  python -m src.cli.prod --input datasets/JIRA.xml --output backlog.csv -v")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check project structure is complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
