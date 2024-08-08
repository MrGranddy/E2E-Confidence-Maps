import argparse
import os

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from data import CMDataset
from model import UNet


import pytorch_lightning as pl
import torch
from torch.optim import Adam


class UNetModule(pl.LightningModule):
    def __init__(self, lr, in_channels, out_channels):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.criterion = torch.nn.MSELoss()
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, conf_maps = batch
        preds = self(images)
        loss = self.criterion(preds, conf_maps)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, conf_maps = batch
        preds = self(images)
        loss = self.criterion(preds, conf_maps)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        val_loss_mean = torch.stack(self.val_losses).mean()
        self.log("val_loss_mean", val_loss_mean, prog_bar=True, logger=True)
        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def objective(trial, args):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model = UNetModule(lr, in_channels=1, out_channels=1)

    # Setup data
    train_dataset = CMDataset(args.images_dir, args.confidence_maps_dir, split="train")
    val_dataset = CMDataset(args.images_dir, args.confidence_maps_dir, split="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # Setup trainer
    logger = TensorBoardLogger(save_dir=args.logdir, name=args.expname)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss")

    trainer = Trainer(
        logger=logger,
        max_epochs=50,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["avg_val_loss"].item()


def main(args):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=20, timeout=3600)

    print(f"Best trial:\n  Value: {study.best_trial.value}\n  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train US image to CM model")
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Directory of the images"
    )
    parser.add_argument(
        "--confidence_maps_dir",
        type=str,
        required=True,
        help="Directory of the confidence maps",
    )
    parser.add_argument(
        "--logdir", type=str, default="logs", help="Directory for saving logs"
    )
    parser.add_argument(
        "--expname", type=str, default="us_image_to_cm", help="Name of the experiment"
    )
    args = parser.parse_args()
    main(args)
