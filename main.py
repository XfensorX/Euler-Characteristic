# TODO: Add Tests
import datetime
import enum
import importlib.util
import os
import shutil

import typer

from utils.configuration import generate_dummy_config, load_config
from utils.result import generate_results

app = typer.Typer()

CAN_CLEAN_DIRS = "results"


class CreationOption(enum.Enum):
    CONFIG = "config"


@app.command()
def create(what: CreationOption, experiment: str):
    experiment_dir = os.path.join("experiments", experiment)
    check_if_exists(experiment_dir)
    generate_dummy_config(
        experiment_dir, ["Custom Model Name 1", "Custom Model Name 2"]
    )


@app.command()
def clean(directory: str):
    """Clean the specified directory."""
    if directory not in CAN_CLEAN_DIRS:
        typer.echo(
            f"Cannot clean '{directory}'. Can only clean: {', '.join(CAN_CLEAN_DIRS)}"
        )
        raise typer.Exit(1)

    if not os.path.exists(directory):
        typer.echo(f"The directory '{directory}' does not exist.")
        raise typer.Exit(1)

    confirm = typer.confirm(
        f"Do you really want to fully remove all contents of the '{directory}' directory?"
    )
    if confirm:
        try:
            shutil.rmtree(directory)
            os.makedirs("results")
            typer.echo(
                f"All contents of the '{directory}' directory have been removed."
            )
        except Exception as e:
            typer.echo(f"An error occurred while trying to clean the directory: {e}")
    else:
        typer.echo("Operation aborted.")


@app.command()
def run(experiment: str, config_file: str = "config.yaml", model: str = None):
    """
    Run experiments.

    :param experiment_dir: Directory containing experiment configuration
    :param config_file: Experiment configuration file
    """

    experiment_dir = os.path.join("experiments", experiment)
    check_if_exists(experiment_dir)

    config = get_config(os.path.join(experiment_dir, config_file))
    module = get_experiment_module(os.path.join(experiment_dir, "train.py"))

    experiment_obj = module.Experiment(config=config, models=module.get_models(config))

    # TODO: Refactoring
    typer.echo(f"Running experiment '{experiment_dir}'\n")
    try:
        results = experiment_obj.run(model=model)
        typer.echo("Generate results...", nl=False)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join("results", f"{current_time}_{experiment}")
        generate_results(experiment_dir, results, results_dir)
        typer.echo(f"\rGenerated results in {results_dir}!      ")
    except KeyError as e:
        typer.echo(e.note)
        raise typer.Exit(code=1)


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
