"""Developer CLI - full pipeline control for development and debugging."""

import sys
from pathlib import Path

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import JIRAPipeline
from src.utils import Config


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file (default: config/default.yaml)",
)
@click.pass_context
def cli(ctx, config):
    """JIRA Ticket Meta Parser - Developer CLI.

    Full control over individual pipeline stages for development and debugging.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(config) if config else Config()


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", type=click.Choice(["auto", "xml", "csv"]), default="auto")
@click.pass_context
def validate(ctx, input_path, format):
    """Validate JIRA export (XML or CSV).

    Runs backbone validation and outputs:
    - backbone_report.csv
    - backbone_summary.json
    """
    click.echo(f"Validating {input_path} (format: {format})")

    config = ctx.obj["config"]
    pipeline = JIRAPipeline(config.to_dict())

    report_df, summary = pipeline._run_validation(input_path, format)

    click.echo(f"\n✓ Validation complete:")
    click.echo(f"  Issues: {summary['issues_count']}")
    click.echo(f"  Unique keys: {summary['unique_keys_count']}")
    click.echo(f"  Errors: {summary['errors']}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", type=click.Choice(["auto", "xml", "csv"]), default="auto")
@click.pass_context
def extract(ctx, input_path, format):
    """Extract variability features from JIRA export.

    Outputs:
    - variability_features.parquet
    """
    click.echo(f"Extracting features from {input_path}")

    config = ctx.obj["config"]
    pipeline = JIRAPipeline(config.to_dict())

    features_df = pipeline._run_feature_extraction(input_path, format)

    click.echo(f"\n✓ Features extracted: {len(features_df)} issues")


@cli.command()
@click.pass_context
def embed(ctx):
    """Generate embeddings from cached features.

    Requires: variability_features.parquet

    Outputs:
    - embeddings.parquet
    """
    click.echo("Generating embeddings")

    config = ctx.obj["config"]
    pipeline = JIRAPipeline(config.to_dict())

    # Load cached features
    features_df = pipeline._load_cached("features", "variability_features")

    # Generate embeddings
    embeddings_df = pipeline._run_embeddings(features_df)

    click.echo(f"\n✓ Embeddings generated: {len(embeddings_df)} vectors")


@cli.command()
@click.pass_context
def index(ctx):
    """Build FAISS index from cached embeddings.

    Requires: embeddings.parquet

    Outputs:
    - faiss_index.ivf
    - faiss_index.keys.npy
    """
    click.echo("Building FAISS index")

    config = ctx.obj["config"]
    pipeline = JIRAPipeline(config.to_dict())

    # Load cached embeddings
    embeddings_df = pipeline._load_cached("embeddings", "embeddings")

    # Build index
    pipeline._run_indexing(embeddings_df)

    stats = pipeline.indexer.get_index_stats()
    click.echo(f"\n✓ FAISS index built:")
    click.echo(f"  Vectors: {stats['n_vectors']}")
    click.echo(f"  Dimension: {stats['dimension']}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--format", type=click.Choice(["auto", "xml", "csv"]), default="auto")
@click.option("--skip-validation", is_flag=True, help="Skip validation stage (use cached)")
@click.option("--skip-training", is_flag=True, help="Skip training (inference only)")
@click.pass_context
def full(ctx, input_path, format, skip_validation, skip_training):
    """Run full pipeline end-to-end.

    Outputs:
    - All intermediate artifacts
    - clean_backlog.csv (final ranked backlog)
    """
    click.echo(f"Running full pipeline on {input_path}")

    config = ctx.obj["config"]
    pipeline = JIRAPipeline(config.to_dict())

    final_df = pipeline.run(
        input_path,
        input_format=format,
        skip_validation=skip_validation,
        skip_training=skip_training,
    )

    click.echo(f"\n✓ Pipeline complete!")
    click.echo(f"  Ranked issues: {len(final_df)}")
    click.echo(f"  Top 10:")
    click.echo(final_df[["rank", "key", "score", "type", "priority"]].head(10).to_string())


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status and cached artifacts."""
    config = ctx.obj["config"]

    artifacts_dir = Path(config["artifacts"]["base_dir"])

    click.echo("Pipeline Status")
    click.echo("=" * 50)

    categories = ["validation", "features", "embeddings", "indices", "models", "backlogs"]

    for category in categories:
        cat_dir = artifacts_dir / category
        if cat_dir.exists():
            files = list(cat_dir.glob("*"))
            click.echo(f"\n{category.upper()}:")
            if files:
                for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    size = f.stat().st_size / 1024  # KB
                    click.echo(f"  - {f.name} ({size:.1f} KB)")
            else:
                click.echo("  (empty)")


@cli.command()
@click.pass_context
def clean(ctx):
    """Clean all cached artifacts."""
    config = ctx.obj["config"]
    artifacts_dir = Path(config["artifacts"]["base_dir"])

    if click.confirm(f"Delete all artifacts in {artifacts_dir}?"):
        import shutil

        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
            artifacts_dir.mkdir(parents=True)
            click.echo("✓ Artifacts cleaned")


def main():
    """Entry point for jira-dev command."""
    cli(obj={})


if __name__ == "__main__":
    main()
