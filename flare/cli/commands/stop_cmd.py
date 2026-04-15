"""flare stop — scale a deployment to zero replicas."""

from __future__ import annotations

import click
from rich.console import Console

from flare.providers.skypilot import get_provider

console = Console()


@click.command()
@click.argument("model_name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt.")
def stop(model_name: str, yes: bool) -> None:
    """Scale a deployment to zero (preserves config, eliminates GPU cost).

    The model will auto-wake when a new request arrives at the gateway.

    \b
    Examples:
      flare stop qwen3-72b
      flare stop llama-3.3-70b --yes
    """
    if not yes:
        click.confirm(
            f"Scale [bold]{model_name}[/bold] to zero replicas?",
            abort=True,
        )

    provider = get_provider()
    with console.status(f"[cyan]Scaling {model_name} to zero...[/cyan]"):
        try:
            provider.stop(model_name)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1)

    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] scaled to zero (SLEEPING). "
                  "The next request will wake it.")
