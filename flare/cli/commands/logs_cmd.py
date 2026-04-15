"""flare logs — stream logs from an active deployment."""

from __future__ import annotations

import click
from rich.console import Console

from flare.providers.skypilot import get_provider

console = Console()


@click.command()
@click.argument("model_name")
@click.option("--replica", default=0, show_default=True, help="Replica index to stream logs from.")
def logs(model_name: str, replica: int) -> None:
    """Stream logs from an active deployment replica.

    \b
    Examples:
      flare logs qwen3-8b
      flare logs llama-3.3-70b --replica 1
    """
    console.print(f"[dim]Streaming logs from[/dim] [bold]{model_name}[/bold] "
                  f"[dim](replica {replica}) — Ctrl+C to stop[/dim]")
    console.print()

    provider = get_provider()
    try:
        for line in provider.stream_logs(model_name, replica=replica):
            console.print(line, highlight=False, markup=False)
    except KeyboardInterrupt:
        console.print("\n[dim]Log streaming stopped.[/dim]")
    except Exception as exc:
        console.print(f"[red]Error streaming logs:[/red] {exc}")
        raise SystemExit(1)
