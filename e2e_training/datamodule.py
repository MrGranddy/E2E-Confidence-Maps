from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from e2e_training.dataset import CMDataset


class CMDataModule(pl.LightningDataModule):
    def __init__(
        self, images_dir: str, confidence_maps_dir: str, batch_size: int
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.confidence_maps_dir = confidence_maps_dir

        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:
            self.train = CMDataset(
                self.images_dir, self.confidence_maps_dir, split="train"
            )
            self.val = CMDataset(self.images_dir, self.confidence_maps_dir, split="val")

        if stage == "test" or stage is None:
            self.test = CMDataset(
                self.images_dir, self.confidence_maps_dir, split="test"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=7,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )
