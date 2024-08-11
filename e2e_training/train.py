import os

import hydra
import lightning.pytorch as pl
from datamodule import CMDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from model import DirectPredictionModule, USAutoEncoderWithConfidenceAsVariance
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:

    if cfg.experimental_setup == "direct_prediction":

        model = DirectPredictionModule(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            lr=cfg.train.lr,
            num_images_to_log=cfg.logger.num_images_to_log,
            use_md_reg=cfg.model.use_md_reg,
            md_reg_weight=cfg.model.md_reg_weight,
        )

    elif cfg.experimental_setup == "autoencoder_with_confidence_as_variance":

        model = USAutoEncoderWithConfidenceAsVariance(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            lr=cfg.train.lr,
            num_images_to_log=cfg.logger.num_images_to_log,
        )

    datamodule = CMDataModule(
        images_dir=cfg.data.images_dir,
        confidence_maps_dir=cfg.data.confidence_maps_dir,
        batch_size=cfg.train.batch_size,
    )

    logger = TensorBoardLogger(cfg.logger.save_dir, name=cfg.logger.name)

    callbacks = [
        ModelCheckpoint(monitor=cfg.callbacks.model_checkpoint.monitor),
        EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
        ),
    ]

    trainer = pl.Trainer(
        logger=logger,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
    )
    trainer.logger.log_hyperparams(cfg.train)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
