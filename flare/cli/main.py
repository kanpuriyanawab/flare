"""Flare CLI entry point."""

import click
from rich.console import Console

from flare.cli.commands.catalog_cmd import catalog
from flare.cli.commands.deploy_cmd import deploy
from flare.cli.commands.init_cmd import init
from flare.cli.commands.apply_cmd import apply
from flare.cli.commands.logs_cmd import logs
from flare.cli.commands.model_cmd import model
from flare.cli.commands.rm_cmd import rm
from flare.cli.commands.stop_cmd import stop
from flare.cli.commands.batch_cmd import batch
from flare.cli.commands.cost_cmd import cost

console = Console()

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="flare-deploy", prog_name="flare")
def cli() -> None:
    """Flare — deploy open-source LLMs on your own cloud infrastructure.

    \b
    Quick start:
      flare init --infra aws          # one-time setup
      flare catalog                   # browse available models
      flare deploy qwen3-8b           # deploy a model
      flare model                     # check deployment status
    """


cli.add_command(init)
cli.add_command(apply)
cli.add_command(model)
cli.add_command(deploy)
cli.add_command(stop)
cli.add_command(rm)
cli.add_command(logs)
cli.add_command(catalog)
cli.add_command(batch)
cli.add_command(cost)


if __name__ == "__main__":
    cli()
