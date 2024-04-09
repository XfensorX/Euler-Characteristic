# TODO: Add Tests
import importlib.util
import os

import typer

from utils.configuration import load_config
from models.VanillaNN import VanillaNN

app = typer.Typer()


@app.command()
def show():
    pass


@app.command()
def run(experiment_dir: str, config_file: str = "config.yaml"):
    """
    Run experiments.

    :param experiment_dir: Directory containing experiment configuration
    :param config_file: Experiment configuration file
    """

    experiment_dir = os.path.join("experiments", experiment_dir)
    check_if_exists(experiment_dir)

    config = get_config(os.path.join(experiment_dir, config_file))
    module = get_experiment_module(os.path.join(experiment_dir, "train.py"))

    experiment = module.Experiment(config=config)

    # TODO: Do not output the confidguration, save some history files for the training
    typer.echo("Running experiment with configuration:")
    typer.echo(str(config) + "\n")
    experiment.run()


def get_config(path: str):
    try:
        config = load_config(path)
    except ValueError as e:
        typer.echo(e.note)
        raise typer.Exit(code=1)

    return config


def check_if_exists(directory: str):
    if not os.path.exists(directory):
        typer.echo(f"'{directory}' not found.")
        raise typer.Exit(code=1)


def get_experiment_module(file_path: str):
    spec = importlib.util.spec_from_file_location("train", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Experiment"):
        typer.echo(f"Error: 'Experiment' class not found in {file_path}")
        raise typer.Exit(code=1)

    return module


if __name__ == "__main__":
    app()
