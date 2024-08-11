from typing import List

import lightning.pytorch as pl
import torch
from unet import UNet
from vis_utils import visualize_predictions


def monotonic_decreasing_regularization(input_tensor):
    """
    Regularization function that rewards monotonically decreasing columns.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W)

    Returns:
        torch.Tensor: A scalar tensor representing the regularization loss.
    """
    # Calculate differences between adjacent elements along the height (H) dimension
    # Use the 'diff' function which calculates the difference between adjacent elements along a specified dimension
    differences = torch.diff(input_tensor, dim=2)

    # Since we want to reward monotonically decreasing values, penalize positive differences
    # Use ReLU as it zeroes out negative differences (which are acceptable for monotonically decreasing)
    # and keeps positive differences (which we want to penalize)
    positive_differences = torch.relu(differences)

    # Sum up all positive differences to get the total penalty
    regularization_loss = positive_differences.sum(dim=(1, 2, 3))

    return regularization_loss.mean()


class DirectPredictionModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lr: float,
        num_images_to_log: int = 4,
        use_md_reg: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters({
                "in_channels": in_channels,
                "out_channels": out_channels,
                "lr": lr,
                "num_images_to_log": num_images_to_log,
                "use_md_reg": use_md_reg,
        })

        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.criterion = torch.nn.MSELoss()
        self.use_md_reg = use_md_reg

        self.train_losses = []
        self.val_losses = []

        self.num_images_to_log = num_images_to_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:

        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        if self.use_md_reg:
            # Calculate the regularization loss
            regularization_loss = monotonic_decreasing_regularization(preds)
            # Add the regularization loss to the main loss
            loss += regularization_loss

        self.log("train_loss", loss.item())
        self.train_losses.append(loss.item())

        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, conf_maps = batch

        preds = self(images)
        loss = self.criterion(preds, conf_maps)

        if self.use_md_reg:
            # Calculate the regularization loss
            regularization_loss = monotonic_decreasing_regularization(preds)
            # Add the regularization loss to the main loss
            loss += regularization_loss

        # Log loss
        self.log("val_loss", loss.item())
        self.log("hp_metric", loss.item(), on_step=False, on_epoch=True)

        self.val_losses.append(loss.item())

        # Save some validation images and corresponding predictions
        if batch_idx == 0:  # Log the first batch in each epoch
            num_images_to_log = min(
                self.num_images_to_log, images.size(0)
            )  # Log up to self.num_images_to_log images

            img_tensors = visualize_predictions(
                images, conf_maps, preds, num_images_to_log
            )

            for i, img_tensor in enumerate(img_tensors):
                self.logger.experiment.add_image(
                    f"val_comparison_{i}", img_tensor, self.current_epoch
                )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class USAutoEncoderWithConfidenceAsVariance(pl.LightningModule):
    def __init__(
        self, in_channels: int, out_channels: int, lr: float, num_images_to_log: int = 4
    ) -> None:
        super().__init__()

        self.save_hyperparameters({
                "in_channels": in_channels,
                "out_channels": out_channels,
                "lr": lr,
                "num_images_to_log": num_images_to_log,
        })

        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.criterion = torch.nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

        self.num_images_to_log = num_images_to_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def negative_log_likelihood(
        self, images: torch.Tensor, pred_means: torch.Tensor, pred_vars: torch.Tensor
    ) -> torch.Tensor:
        # Calculate the negative log likelihood loss
        loss_for_each_pixel = -0.5 * ((images - pred_means) ** 2) / pred_vars
        loss = loss_for_each_pixel.mean()
        return loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:

        images, conf_maps = batch

        pred_means = self(images)
        pred_vars = conf_maps

        # minimize -log(p(images | means, vars)) for pred_mean
        loss = self.negative_log_likelihood(images, pred_means, pred_vars)

        self.log("train_loss", loss.item())
        self.train_losses.append(loss.item())

        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        images, conf_maps = batch

        pred_means = self(images)
        pred_vars = conf_maps

        # minimize -log(p(images | means, vars)) for pred_mean
        loss = self.negative_log_likelihood(images, pred_means, pred_vars)

        # Log loss
        self.log("val_loss", loss.item())
        self.log("hp_metric", loss.item(), on_step=False, on_epoch=True)

        self.val_losses.append(loss.item())

        # Save some validation images and corresponding predictions
        if batch_idx == 0:
            num_images_to_log = min(self.num_images_to_log, images.size(0))

            img_tensors = visualize_predictions(
                images, conf_maps, pred_means, num_images_to_log
            )

            for i, img_tensor in enumerate(img_tensors):
                self.logger.experiment.add_image(
                    f"val_comparison_{i}", img_tensor, self.current_epoch
                )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
