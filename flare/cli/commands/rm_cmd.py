"""flare rm — tear down a deployment completely."""

from __future__ import annotations

import click
from rich.console import Console

from flare.providers.skypilot import get_provider

console = Console()


@click.command()
@click.argument("model_name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt.")
def rm(model_name: str, yes: bool) -> None:
    """Tear down a deployment completely (removes all cloud resources).

    Unlike `flare stop`, this deletes the deployment config from SkyPilot.
    Use `flare deploy` again to redeploy.

    \b
    Examples:
      flare rm qwen3-72b
      flare rm llama-3.3-70b --yes
    """
    if not yes:
        click.confirm(
            f"[bold red]Permanently remove[/bold red] deployment [bold]{model_name}[/bold]? "
            "This will terminate all cloud resources.",
            abort=True,
        )

    provider = get_provider()
    with console.status(f"[red]Tearing down {model_name}...[/red]"):
        try:
            provider.teardown(model_name)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1)

    console.print(f"[green]✓[/green] [bold]{model_name}[/bold] removed. "
                  "All cloud resources have been terminated.")
