"""Production CLI - simplified interface for end users."""

import sys
from pathlib import Path

import click

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import JIRAPipeline
from src.utils import Config


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to JIRA export file (XML or CSV)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    default=None,
    help="Path to output CSV (default: artifacts/backlogs/clean_backlog.csv)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file (default: config/default.yaml)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["auto", "xml", "csv"]),
    default="auto",
    help="Input format (default: auto-detect)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def main(input_path, output_path, config, format, verbose):
    """JIRA Ticket Meta Parser - Production CLI.

    Validates and ranks JIRA tickets in a single command.

    Example:
        jira-validate --input JIRA.xml --output backlog.csv
    """
    if verbose:
        click.echo("JIRA Ticket Meta Parser v1.0.0")
        click.echo("=" * 50)

    # Load config
    cfg = Config(config) if config else Config()

    # Initialize pipeline
    if verbose:
        click.echo(f"\n[1/4] Loading pipeline...")
    pipeline = JIRAPipeline(cfg.to_dict())

    # Run pipeline
    if verbose:
        click.echo(f"[2/4] Processing {input_path}...")

    try:
        final_df = pipeline.run(
            input_path,
            input_format=format,
            skip_validation=False,
            skip_training=True,  # Production mode: inference only
        )

        # Save to custom output if specified
        if output_path:
            if verbose:
                click.echo(f"[3/4] Saving to {output_path}...")
            final_df.to_csv(output_path, index=False)
            result_path = output_path
        else:
            # Use default artifact path
            result_path = (
                Path(cfg["artifacts"]["base_dir"]) / "backlogs" / "clean_backlog.csv"
            )

        # Summary
        if verbose:
            click.echo(f"[4/4] Complete!\n")

        click.echo(f"✓ Successfully ranked {len(final_df)} issues")
        click.echo(f"  Output: {result_path}")

        if verbose:
            click.echo(f"\nTop 10 issues:")
            click.echo(
                final_df[["rank", "key", "score", "type", "priority"]]
                .head(10)
                .to_string(index=False)
            )

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
