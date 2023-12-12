import os

import click
from cli.cli_base import CLIException
from model_maker.file_model_maker import FileModelMaker

FLYMYAI_PROJECT_DIR = os.getenv("FLYMYAI_PROJECT_DIR")


@click.command
@click.option("--input-json")
@click.option("--project_name", "-p")
@click.option("--dir", "-d", help="Explicit path to project modules")
def cli(input_json, project_name, dir_name=None):
    projects_directory = dir_name or FLYMYAI_PROJECT_DIR
    if not projects_directory:
        raise CLIException("FLYMYAI_PROJECT_DIR (environment var) or --dir required")
    model_maker = FileModelMaker(
        input_json,
    )
    model_maker.generate_model_py()


if __name__ == "__main__":
    cli()
