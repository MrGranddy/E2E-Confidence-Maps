import argparse
import os

import optuna
import lightning.pytorch as pl

from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from datamodule import CMDataModule
from model import DirectPredictionModule

BATCHSIZE = 32
N_TRIALS = 20
EPOCHS = 30
PERCENT_TRAIN_EXAMPLES = 1.0
PERCENT_VALID_EXAMPLES = 1.0
DIR = os.getcwd()

def objective(trial: optuna.trial.Trial) -> float:

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = DirectPredictionModule(in_channels=1, out_channels=1, lr=lr)
    datamodule = CMDataModule(
        images_dir="../CM_datasets/images",
        confidence_maps_dir="../CM_datasets/ultranerf",
        batch_size=BATCHSIZE,
    )

    logger = TensorBoardLogger("logs", name="DirectPredictionModule")

    callbacks = [
        # ModelCheckpoint(monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=3),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        limit_train_batches=PERCENT_TRAIN_EXAMPLES,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
    )
    hyperparameters = dict(lr=lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))