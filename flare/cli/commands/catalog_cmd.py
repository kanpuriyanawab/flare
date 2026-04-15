"""flare catalog — browse the model registry."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich import box

from flare.registry.loader import get_registry

console = Console()


@click.command()
@click.option("--family", default=None, help="Filter by model family (e.g. qwen, llama).")
@click.option("--tag", default=None, help="Filter by tag (e.g. gguf, reasoning).")
@click.option("--capability", default=None, help="Filter by capability (e.g. vision, reasoning).")
@click.option("--engine", default=None, help="Filter by serving engine (vllm, llama-cpp, sglang).")
def catalog(
    family: str | None,
    tag: str | None,
    capability: str | None,
    engine: str | None,
) -> None:
    """Browse the Flare model registry.

    \b
    Examples:
      flare catalog
      flare catalog --family qwen
      flare catalog --tag gguf
      flare catalog --capability reasoning
    """
    registry = get_registry()
    models = registry.search(family=family, tag=tag, capability=capability, engine=engine)

    if not models:
        console.print("[yellow]No models matched the given filters.[/yellow]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"[bold]Flare Model Registry[/bold] ([dim]{len(models)} models[/dim])",
        title_style="bold",
        expand=True,
    )
    table.add_column("Name", style="bold white", min_width=22)
    table.add_column("Display Name", min_width=28)
    table.add_column("Family", style="cyan", min_width=10)
    table.add_column("Engine", style="magenta", min_width=9)
    table.add_column("GPU (Recommended)", style="yellow", min_width=22)
    table.add_column("Mem", style="dim", min_width=6, justify="right")
    table.add_column("Tags", style="dim", min_width=20)

    for spec in models:
        gpu_str = ", ".join(spec.gpus.recommended[:2])
        tag_str = ", ".join(spec.tags[:4])
        mem_str = f"{int(spec.memory_gb)}GB"
        engine_str = spec.serving.engine.value

        if spec.is_gguf:
            name_display = f"[dim]{spec.name}[/dim]"
            engine_str = f"[dim]{engine_str}[/dim]"
        else:
            name_display = spec.name

        table.add_row(
            name_display,
            spec.display_name,
            spec.family,
            engine_str,
            gpu_str,
            mem_str,
            tag_str,
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Deploy a model:[/dim] [bold]flare deploy <name>[/bold]   "
        "[dim]Get details:[/dim] [bold]flare catalog --family <family>[/bold]"
    )
