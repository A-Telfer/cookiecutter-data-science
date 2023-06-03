# -*- coding: utf-8 -*-
import logging
import random
from pathlib import Path

import click
import mlflow

# import numpy as np
# import torch
from dotenv import find_dotenv, load_dotenv


@click.command(help="Train a Model")
@click.option("--training_data", default="data/processed", type=click.Path())
@click.option("--epochs", type=click.INT, default=10, help="Number of training epochs.")
@click.option("--learning_rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--seed", type=click.INT, default=97531, help="Seed random number generators.")
def main(**kwargs):
    """Train a model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training model")

    training_data = kwargs['training_data']
    epochs = kwargs['epochs']
    learning_rate = kwargs['learning_rate']
    seed = kwargs['seed']

    # Seeding
    logger.info(f"Setting random seed {seed}")
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # g = torch.Generator()
    # g.manual_seed(seed)

    with mlflow.start_run() as active_run:
        mlflow.log_params(kwargs)
        # Load datasets
        # Train model

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
