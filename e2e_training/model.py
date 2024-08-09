from typing import List

import lightning.pytorch as pl
import torch

from unet import UNet


class DirectPredictionModule(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr: float) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.criterion = torch.nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:

        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.train_losses.append(loss)

        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("hp_metric", loss, on_epoch=True, on_step=False, logger=True)

        self.val_losses.append(loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
