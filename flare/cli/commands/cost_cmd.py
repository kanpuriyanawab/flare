"""flare cost — show cost tracking and savings (Phase 2)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.table import Table
from rich import box

from flare.core.constants import GPU_HOURLY_COSTS, OPENAI_TOKEN_PRICING, DEFAULT_OPENAI_COMPARISON_MODEL

console = Console()

_PERIOD_MAP = {
    "1d": 1, "7d": 7, "30d": 30, "90d": 90,
}


@click.command()
@click.option(
    "--period",
    default="7d",
    type=click.Choice(list(_PERIOD_MAP.keys())),
    show_default=True,
    help="Time period for cost report.",
)
@click.option("--model", "model_filter", default=None, help="Filter by model name.")
def cost(period: str, model_filter: str | None) -> None:
    """Show GPU cost breakdown and savings vs OpenAI.

    \b
    Examples:
      flare cost
      flare cost --period 30d
      flare cost --period 7d --model qwen3-72b
    """
    asyncio.run(_cost_async(period, model_filter))


async def _cost_async(period: str, model_filter: str | None) -> None:
    days = _PERIOD_MAP[period]
    since = datetime.utcnow() - timedelta(days=days)

    try:
        from flare.gateway.queue.sqlite_queue import get_cost_records
        records = await get_cost_records(since=since, model_filter=model_filter)
    except Exception:
        records = []

    if not records:
        console.print(f"[dim]No cost data found for the last {period}.[/dim]")
        console.print("[dim]Cost tracking starts once the gateway has handled requests.[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title=f"[bold]Cost Report[/bold] ([dim]last {period}[/dim])",
        expand=True,
    )
    table.add_column("Model", style="bold white", min_width=22)
    table.add_column("Mode", min_width=10)
    table.add_column("GPU", style="yellow", min_width=12)
    table.add_column("Hours", justify="right", min_width=8)
    table.add_column("Est. Cost", style="green", justify="right", min_width=12)
    table.add_column(f"vs {DEFAULT_OPENAI_COMPARISON_MODEL}", style="cyan", justify="right", min_width=16)

    total_cost = 0.0
    total_openai_equiv = 0.0

    openai_price = OPENAI_TOKEN_PRICING[DEFAULT_OPENAI_COMPARISON_MODEL]

    for rec in records:
        gpu_type = rec.get("gpu_type", "A100").split(":")[0]
        gpu_count = int(rec.get("gpu_count", 1))
        total_seconds = float(rec.get("total_seconds", 0))
        hours = total_seconds / 3600
        hourly = GPU_HOURLY_COSTS.get(gpu_type, 3.0) * gpu_count
        est_cost = hourly * hours

        # Rough token-based comparison (assume 1M tokens/hour per GPU)
        est_tokens_m = hours * gpu_count
        openai_equiv = est_tokens_m * openai_price

        savings_pct = ((openai_equiv - est_cost) / openai_equiv * 100) if openai_equiv > 0 else 0
        savings_str = (
            f"${openai_equiv:.2f} [dim](saved {savings_pct:.0f}%)[/dim]"
            if openai_equiv > est_cost
            else f"${openai_equiv:.2f}"
        )

        total_cost += est_cost
        total_openai_equiv += openai_equiv

        table.add_row(
            rec.get("model_name", "—"),
            rec.get("mode", "on-demand"),
            rec.get("gpu_type", "—"),
            f"{hours:.1f}h",
            f"${est_cost:.2f}",
            savings_str,
        )

    console.print()
    console.print(table)

    total_savings = total_openai_equiv - total_cost
    console.print()
    console.print(
        f"[bold]Total:[/bold] [green]${total_cost:.2f}[/green] "
        f"vs [dim]${total_openai_equiv:.2f} on {DEFAULT_OPENAI_COMPARISON_MODEL}[/dim] "
        f"— [cyan]saved ${total_savings:.2f} ({total_savings/total_openai_equiv*100:.0f}%)[/cyan]"
        if total_openai_equiv > 0
        else f"[bold]Total:[/bold] [green]${total_cost:.2f}[/green]"
    )
