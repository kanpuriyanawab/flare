"""flare batch — submit and monitor batch inference jobs (Phase 2)."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich import box

from flare.core.config import BatchJobSpec
from flare.core.exceptions import ModelNotFoundError
from flare.providers.skypilot import get_provider
from flare.registry.loader import get_registry

console = Console()


@click.group()
def batch() -> None:
    """Batch inference jobs on spot instances (cost-optimized)."""


@batch.command()
@click.argument("model_name")
@click.option("--input", "input_path", required=True, help="Input JSONL path (local or s3:// / gs://).")
@click.option("--output", "output_path", required=True, help="Output path (local or s3:// / gs://).")
@click.option("--gpu", default=None, help="GPU override (e.g. A100:4).")
@click.option("--no-spot", is_flag=True, help="Use on-demand instances (more expensive).")
def submit(
    model_name: str,
    input_path: str,
    output_path: str,
    gpu: str | None,
    no_spot: bool,
) -> None:
    """Submit a batch inference job.

    \b
    Input format (JSONL, one request per line):
      {"id": "req1", "prompt": "What is the capital of France?"}
      {"id": "req2", "messages": [{"role": "user", "content": "Hello"}]}

    \b
    Examples:
      flare batch submit qwen3-8b --input inputs.jsonl --output s3://my-bucket/results/
      flare batch submit llama-3.3-70b --input gs://bucket/in.jsonl --output gs://bucket/out/
    """
    registry = get_registry()
    try:
        spec = registry.get(model_name)
    except ModelNotFoundError:
        console.print(f"[red]Model not found:[/red] {model_name}")
        raise SystemExit(1)

    job = BatchJobSpec(
        model_name=model_name,
        input_path=input_path,
        output_path=output_path,
        gpu=gpu,
        use_spot=not no_spot,
    )

    provider = get_provider()
    with console.status(f"[cyan]Submitting batch job for {model_name}...[/cyan]"):
        try:
            job_id = provider.submit_batch_job(spec, job)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise SystemExit(1)

    console.print(f"[green]✓[/green] Batch job submitted: [bold]{job_id}[/bold]")
    console.print(f"  Input:  {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Spot:   {'yes' if not no_spot else 'no'}")
    console.print(f"\nCheck status with: [bold]flare batch status[/bold]")


@batch.command()
def status() -> None:
    """List all batch jobs and their status.

    \b
    Examples:
      flare batch status
    """
    provider = get_provider()
    jobs = provider.list_batch_jobs()

    if not jobs:
        console.print("[dim]No batch jobs found.[/dim]")
        return

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="[bold]Batch Jobs[/bold]",
    )
    table.add_column("Job ID", style="bold white", min_width=30)
    table.add_column("Model", min_width=18)
    table.add_column("Status", min_width=12)
    table.add_column("Submitted", min_width=20)
    table.add_column("Output", style="dim")

    for job in jobs:
        job_id = job.get("job_id") or job.get("job_name", "—")
        status_val = str(job.get("status", "UNKNOWN"))
        color = {
            "RUNNING": "yellow",
            "SUCCEEDED": "green",
            "FAILED": "red",
            "PENDING": "cyan",
        }.get(status_val.upper(), "white")

        table.add_row(
            job_id,
            job.get("task_name", "—"),
            f"[{color}]{status_val}[/{color}]",
            str(job.get("submitted_at", "—")),
            job.get("output_path", "—"),
        )

    console.print()
    console.print(table)
