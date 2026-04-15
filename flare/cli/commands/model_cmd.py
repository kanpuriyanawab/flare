"""flare model — list all active deployments."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich import box

from flare.core.state import DeploymentState
from flare.providers.skypilot import get_provider

console = Console()


@click.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def model(as_json: bool) -> None:
    """List all model deployments and their current status.

    \b
    Examples:
      flare model
      flare model --json
    """
    provider = get_provider()
    deployments = provider.list_deployments()

    if as_json:
        import json
        data = [d.model_dump() for d in deployments]
        console.print_json(json.dumps(data))
        return

    if not deployments:
        console.print("[dim]No active deployments. Run [bold]flare deploy <model>[/bold] to get started.[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="[bold]Active Deployments[/bold]",
        expand=True,
    )
    table.add_column("Model", style="bold white", min_width=22)
    table.add_column("State", min_width=14)
    table.add_column("Replicas", justify="center", min_width=10)
    table.add_column("GPU", style="yellow", min_width=12)
    table.add_column("Cost/hr", style="green", justify="right", min_width=10)
    table.add_column("Endpoint", style="dim", min_width=35)

    for d in deployments:
        state = DeploymentState(d.state) if d.state in [s.value for s in DeploymentState] else DeploymentState.UNKNOWN
        state_str = f"[{state.display_color}]{state.value}[/{state.display_color}]"

        replicas_str = f"{d.replicas_ready}/{d.replicas_total}"
        cost_str = f"~${d.cost_per_hour:.2f}" if d.cost_per_hour else "[dim]$0.00[/dim]"
        endpoint_str = d.endpoint or "[dim]—[/dim]"
        gpu_str = d.gpu or "[dim]—[/dim]"

        table.add_row(
            d.name,
            state_str,
            replicas_str,
            gpu_str,
            cost_str,
            endpoint_str,
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Tip:[/dim] Requests to sleeping models auto-queue and wake them. "
        "[bold]flare stop <model>[/bold] to scale to zero manually."
    )
