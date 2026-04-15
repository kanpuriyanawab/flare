"""flare deploy — deploy a single model from the registry."""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from flare.core.config import DeploymentEntry, FlareConfig, GlobalDefaults
from flare.core.constants import FLARE_CONFIG_PATH
from flare.core.exceptions import ModelNotFoundError
from flare.providers.skypilot import get_provider
from flare.registry.loader import get_registry

console = Console()


@click.command()
@click.argument("model_name")
@click.option("--gpu", default=None, help="GPU spec override, e.g. A100:4")
@click.option("--min-replicas", default=0, show_default=True, help="Minimum replicas.")
@click.option("--max-replicas", default=3, show_default=True, help="Maximum replicas.")
@click.option("--idle-timeout", default="15m", show_default=True, help="Scale-to-zero timeout.")
@click.option("--infra", default=None, help="Cloud provider (overrides config).")
def deploy(
    model_name: str,
    gpu: str | None,
    min_replicas: int,
    max_replicas: int,
    idle_timeout: str,
    infra: str | None,
) -> None:
    """Deploy a model from the registry.

    \b
    Examples:
      flare deploy qwen3-8b
      flare deploy llama-3.3-70b --gpu A100:4 --max-replicas 2
      flare deploy llama-3-8b-q4km --gpu L4:1 --idle-timeout 10m
    """
    registry = get_registry()

    try:
        spec = registry.get(model_name)
    except ModelNotFoundError:
        console.print(f"[red]Model not found:[/red] [bold]{model_name}[/bold]")
        console.print("Run [bold]flare catalog[/bold] to see available models.")
        raise SystemExit(1)

    entry = DeploymentEntry(
        name=model_name,
        mode="on-demand",
        gpu=gpu,
        idle_timeout=idle_timeout,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )
    defaults = GlobalDefaults()

    console.print()
    console.print(Panel.fit(
        f"[bold]Deploying:[/bold] {spec.display_name}\n"
        f"  GPU:          [cyan]{gpu or spec.default_gpu}[/cyan]\n"
        f"  Engine:       [magenta]{spec.serving.engine.value}[/magenta]\n"
        f"  Replicas:     [yellow]{min_replicas} → {max_replicas}[/yellow]\n"
        f"  Idle timeout: [dim]{idle_timeout}[/dim]\n"
        f"  Est. startup: [dim]~{spec.startup_time_seconds}s[/dim]",
        border_style="cyan",
        title="[bold cyan]flare deploy[/bold cyan]",
    ))
    console.print()

    provider = get_provider()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]Launching {model_name} via SkyPilot...", total=None)
        try:
            service_name = provider.deploy(spec, entry, defaults)
            progress.update(
                task,
                description=f"[green]✓[/green] Deployment submitted: [bold]{service_name}[/bold]",
            )
            progress.stop_task(task)
        except Exception as exc:
            progress.update(task, description=f"[red]✗[/red] Deploy failed: {exc}")
            progress.stop_task(task)
            raise SystemExit(1)

    console.print()
    console.print(
        f"[green]Deployment launched![/green] Run [bold]flare model[/bold] to check status.\n"
        f"Endpoint will be available at the gateway once RUNNING."
    )
