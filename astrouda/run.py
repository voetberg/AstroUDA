"""
Train models for open/closed/partial/partial-open domain adaptation problems
"""
import os
from astrouda.models.train import TrainDA
import click
from astrouda.utils.utils import load_config

@click.command()
@click.argument("config", help="Path to the configuration file, yaml file containing dataset and model parameters")
@click.pass_context
def run(ctx, config: str): 
    if config is not None: 
        ctx.obj['config'] = load_config(config)


@run.command()
@click.pass_context
def train(ctx):
    "Train a model using the provided configuration and data."
    TrainDA(ctx.obj['config'])()

@run.command()
@click.pass_context
def test(ctx):
    """
    Run inference on the trained model.
    """
    train_da = TrainDA(ctx.obj['config'])
    train_da.test()