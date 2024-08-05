"""The app is used through this module."""

import click

from . import main

main()


@click.command()
@click.option(
    "--loader",
    type=click.Choice(("DIR", "CSV", "JSON", "MARKDOWN", "HTML", "PDF")),
    default="DIR",
    help="Loader type",
)
@click.option(
    "--chunker",
    type=click.Choice(
        (
            "CHARACTER",
            "RECURSIVE",
            "HTMLHEADER",
            "HTMLSECTION",
            "MARKDOWN",
            "SEMANTIC",
            "TOKEN",
        ),
    ),
    default="RECURSIVE",
    help="Chunker type",
)
@click.option(
    "--persister",
    type=click.Choice(["memory"]),
    default="memory",
    help="Persister type",
)
@click.option(
    "--retriever",
    type=click.Choice(["simple"]),
    default="simple",
    help="Retriever type",
)
@click.option(
    "--generator",
    type=click.Choice(["simple"]),
    default="simple",
    help="Generator type",
)
@click.option(
    "--evaluators",
    multiple=True,
    type=click.Choice(["simple"]),
    default=["simple"],
    help="Evaluators",
)
def configure_and_run(
    loader, chunker, persister, retriever, generator, evaluators
):
    pass
