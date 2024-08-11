import argparse
import os

import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from datamodule import CMDataModule
from model import DirectPredictionModule

BATCHSIZE = 32
LR = 1e-4
EPOCHS = 30
PERCENT_TRAIN_EXAMPLES = 1.0
PERCENT_VALID_EXAMPLES = 1.0

DIR = "G:/"
IMAGES_DIR = os.path.join(DIR, "CM_datasets/images")
CONFIDENCE_MAPS_DIR = os.path.join(DIR, "CM_datasets/ultranerf")

def train() -> None:

    model = DirectPredictionModule(in_channels=1, out_channels=1, lr=LR)
    datamodule = CMDataModule(
        images_dir=IMAGES_DIR,
        confidence_maps_dir=CONFIDENCE_MAPS_DIR,
        batch_size=BATCHSIZE,
    )

    logger = TensorBoardLogger("logs", name="DirectPredictionModuleWithMDReg")

    callbacks = [
        ModelCheckpoint(monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=3)
    ]

    trainer = pl.Trainer(
        logger=logger,
        limit_train_batches=PERCENT_TRAIN_EXAMPLES,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=True,
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
    )
    hyperparameters = dict(lr=LR, batch_size=BATCHSIZE)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":

    train()