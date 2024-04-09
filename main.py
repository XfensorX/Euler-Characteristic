import argparse
import importlib.util
import os

import typer

from utils.configuration import load_config
from models.VanillaNN import VanillaNN

app = typer.Typer()


@app.command()
def run_experiment(experiment_dir: str, config_file: str = "config.yaml"):
    """
    Run experiments.

    :param experiment_dir: Directory containing experiment configuration
    :param config_file: Experiment configuration file
    """

    experiment_dir = os.path.join("experiments", experiment_dir)

    if not os.path.exists(experiment_dir):
        typer.echo(f"Experiment directory '{experiment_dir}' not found.")
        raise typer.Exit(code=1)

    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(experiment_dir, "train.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "Experiment"):
        try:
            config = load_config(experiment_dir, config_file)
        except TypeError as e:
            typer.echo(
                f"Wrong Format in the configuration of {experiment_dir}. The following configurations are missing:\n"
            )
            typer.echo(":\n".join(str(e).split("'")[1::2] + [""]))
            raise typer.Exit(code=1)
        except ValueError as e:
            typer.echo(str(e))
            typer.echo(str(e.note))
            raise typer.Exit(code=1)

        # TODO: Add MOdels configuration
        experiment = module.Experiment(
            config=config,
            model=VanillaNN(
                config.img_x_size * config.img_y_size, 1, [20], flatten_input=True
            ),
        )
        # TODO: Do not output the confidguration, save some history files for the training
        typer.echo("Running experiment with configuration:")
        typer.echo(config)
        typer.echo("")
        experiment.run()
    else:
        typer.echo(f"Error: 'Experiment' class not found in {experiment_dir}/train.py")
        raise typer.Exit(code=1)


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("experiment", help="Name of the experiment to run")
    args = parser.parse_args()

    run_experiment(args.experiment)


if __name__ == "__main__":
    app()
