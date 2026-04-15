"""flare init — one-time infrastructure setup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from flare.core.config import InfraProvider
from flare.core.constants import FLARE_CONFIG_PATH, FLARE_STATE_DIR

console = Console()


@click.command()
@click.option(
    "--infra",
    type=click.Choice([p.value for p in InfraProvider], case_sensitive=False),
    required=True,
    help="Cloud provider to deploy on.",
)
@click.option("--region", default=None, help="Preferred cloud region.")
@click.option("--skip-skypilot", is_flag=True, help="Skip SkyPilot installation check.")
def init(infra: str, region: str | None, skip_skypilot: bool) -> None:
    """One-time setup: validate credentials and configure Flare.

    \b
    Examples:
      flare init --infra aws
      flare init --infra gcp --region us-central1
      flare init --infra kubernetes
    """
    console.print(Panel.fit("[bold cyan]Flare[/bold cyan] — initializing infrastructure", border_style="cyan"))
    console.print()

    # Create state directory
    FLARE_STATE_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("Checking Python version", _check_python),
        ("Installing/verifying SkyPilot", lambda: _check_skypilot(infra, skip_skypilot)),
        ("Validating cloud credentials", lambda: _validate_credentials(infra)),
        ("Writing Flare config", lambda: _write_config(infra, region)),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        for description, fn in steps:
            task = progress.add_task(f"[cyan]{description}...", total=None)
            try:
                fn()
                progress.update(task, description=f"[green]✓[/green] {description}")
                progress.stop_task(task)
            except Exception as exc:
                progress.update(task, description=f"[red]✗[/red] {description}: {exc}")
                progress.stop_task(task)
                console.print(f"\n[red]Error:[/red] {exc}")
                sys.exit(1)

    console.print()
    console.print(Panel.fit(
        f"[bold green]Flare initialized![/bold green]\n\n"
        f"  Infra:   [cyan]{infra}[/cyan]\n"
        f"  Region:  [cyan]{region or 'auto-select'}[/cyan]\n"
        f"  Config:  [dim]{FLARE_CONFIG_PATH}[/dim]\n\n"
        "Run [bold]flare catalog[/bold] to browse models, then\n"
        "[bold]flare deploy <model>[/bold] to get started.",
        border_style="green",
    ))


def _check_python() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python 3.9+ required (found {sys.version})")


def _check_skypilot(infra: str, skip: bool) -> None:
    if skip:
        return
    try:
        import sky  # noqa: F401
    except ImportError:
        console.print(f"\n  [yellow]SkyPilot not found. Installing skypilot[{infra}]...[/yellow]")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", f"skypilot[{infra}]"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed:\n{result.stderr}")


def _validate_credentials(infra: str) -> None:
    validators = {
        "aws": _check_aws,
        "gcp": _check_gcp,
        "azure": _check_azure,
        "kubernetes": _check_kubernetes,
        "lambda": lambda: None,
        "runpod": lambda: None,
    }
    validators.get(infra, lambda: None)()


def _check_aws() -> None:
    try:
        import boto3
        sts = boto3.client("sts")
        sts.get_caller_identity()
    except Exception as exc:
        raise RuntimeError(
            f"AWS credentials not configured: {exc}\n"
            "  Run: aws configure"
        ) from exc


def _check_gcp() -> None:
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "GCP credentials not configured.\n"
            "  Run: gcloud auth application-default login"
        )


def _check_azure() -> None:
    result = subprocess.run(
        ["az", "account", "show"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Azure credentials not configured.\n"
            "  Run: az login"
        )


def _check_kubernetes() -> None:
    result = subprocess.run(
        ["kubectl", "cluster-info"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "kubectl not configured or cluster unreachable.\n"
            "  Ensure KUBECONFIG is set and cluster is running."
        )


def _write_config(infra: str, region: str | None) -> None:
    config = {"infra": infra}
    if region:
        config["region"] = region

    FLARE_CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False))
