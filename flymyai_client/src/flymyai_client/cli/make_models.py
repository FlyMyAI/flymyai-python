import argparse
import os

import click

from flymyai_client.src.vli_client.cli.cli_base import CLIException

print(
            f"""
                 - 
                 - flymyai make_models --input-json=... - generate with json file
            """
        )


@click.command(
    help="""
        - flymyai make_models --username ... --apikey ... - generate with project schema
        - flymyai make_models --env - generate with environment variables: FLYMYAI_APIKEY, FLYMYAI_USERNAME 
    """
)
@click.option('--username', '-u', help='Your username')
@click.option('--apikey', '-a', help='API Key')
@click.option('--project_name', '-p')
def cli(username=None, apikey=None):
    if not username:
        username = os.getenv('FLYMYAI_USERNAME')
    if not apikey:
        apikey = os.getenv('FLYMYAI_APIKEY')
    if not any([username, apikey]):
        raise CLIException("Username or apikey were not provided")


if __name__ == '__main__':
    cli()
