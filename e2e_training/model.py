from typing import List

import lightning.pytorch as pl
import torch
import torchvision

from unet import UNet

from vis_utils import visualize_predictions


class DirectPredictionModule(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr: float, num_images_to_log: int = 4) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.criterion = torch.nn.MSELoss()

        self.train_losses = []
        self.val_losses = []
        
        self.num_images_to_log = num_images_to_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:

        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        self.log("train_loss", loss.item())
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        # Log loss
        self.log("val_loss", loss.item())
        self.log("hp_metric", loss.item(), on_step=False, on_epoch=True)

        self.val_losses.append(loss.item())

        # Save some validation images and corresponding predictions
        if batch_idx == 0:  # Log the first batch in each epoch
            num_images_to_log = min(self.num_images_to_log, images.size(0))  # Log up to self.num_images_to_log images

            img_tensors = visualize_predictions(images, conf_maps, preds, num_images_to_log)

            for i, img_tensor in enumerate(img_tensors):
                self.logger.experiment.add_image(f'val_comparison_{i}', img_tensor, self.current_epoch)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
