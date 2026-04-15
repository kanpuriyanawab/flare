"""flare apply — reconcile desired state from models.yaml."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich import box

from flare.core.config import FlareConfig
from flare.core.exceptions import ModelNotFoundError
from flare.providers.skypilot import get_provider
from flare.registry.loader import get_registry

console = Console()


@click.command()
@click.argument("config_file", default="models.yaml", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would change without applying.")
def apply(config_file: str, dry_run: bool) -> None:
    """Reconcile the desired model deployment state from a config file.

    Deploys new models, updates existing ones, and removes models not in the config.

    \b
    Examples:
      flare apply
      flare apply models.yaml
      flare apply models.yaml --dry-run
    """
    config = FlareConfig.from_yaml(config_file)
    registry = get_registry()
    provider = get_provider()

    # Validate all models exist in registry
    errors = []
    for entry in config.models:
        try:
            registry.get(entry.name)
        except ModelNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise SystemExit(1)

    # Get current deployments
    current = {d.name: d for d in provider.list_deployments()}
    desired = {e.name for e in config.models}
    to_remove = set(current.keys()) - desired
    to_deploy = desired - set(current.keys())
    to_update = desired & set(current.keys())

    # Show plan
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Model", style="bold white")
    table.add_column("Action", min_width=10)
    table.add_column("GPU")
    table.add_column("Replicas")

    for name in sorted(to_deploy):
        entry = next(e for e in config.models if e.name == name)
        spec = registry.get(name)
        gpu = entry.gpu or spec.default_gpu
        min_r = entry.resolved_min_replicas(config.defaults)
        max_r = entry.resolved_max_replicas(config.defaults)
        table.add_row(name, "[green]+ deploy[/green]", gpu, f"{min_r}→{max_r}")

    for name in sorted(to_update):
        table.add_row(name, "[yellow]~ update[/yellow]", "", "")

    for name in sorted(to_remove):
        table.add_row(name, "[red]- remove[/red]", "", "")

    console.print()
    console.print(table)

    if dry_run:
        console.print("[dim]Dry run — no changes applied.[/dim]")
        return

    if not (to_deploy or to_remove):
        console.print("[green]✓[/green] Everything is up to date.")
        return

    # Apply changes
    for name in sorted(to_remove):
        with console.status(f"[red]Removing {name}...[/red]"):
            provider.teardown(name)
        console.print(f"[green]✓[/green] Removed: [bold]{name}[/bold]")

    for name in sorted(to_deploy):
        entry = next(e for e in config.models if e.name == name)
        spec = registry.get(name)
        with console.status(f"[cyan]Deploying {name}...[/cyan]"):
            provider.deploy(spec, entry, config.defaults)
        console.print(f"[green]✓[/green] Deployed: [bold]{name}[/bold]")

    console.print(f"\n[green]Apply complete.[/green] Run [bold]flare model[/bold] to check status.")
