"""
Train models for open/closed/partial/partial-open domain adaptation problems
"""
import os
import click
from astrouda.utils import load_config

@click.command()
@click.option("--config", "-c", help="Path to the configuration file, yaml file containing dataset and model parameters")
@click.option("--data", "-d", default=f"{os.getcwd().rstrip('/')}/data", help="Directory containing the dataset files")
@click.option("--output", "-o", default=f"{os.getcwd().rstrip('/')}/results", help="Folder to save training history and checkpoints")
@click.pass_context
def run(ctx, config: str, data: str, output: str): 
    if config is not None: 
        ctx.obj['config'] = load_config(config)

        data_cfg = ctx.obj['config']['data']

        data = ""  # Load using the specified data loader
        ctx.obj['data'] = data

    ctx.obj['output'] = output


@run.command()
@click.pass_context
def train(ctx):
    "Train a model using the provided configuration and data."

@run.command()
@click.pass_context
def inference(ctx):
    """
    Run inference on the trained model.
    """
    pass
