import os

import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import CMDataset
from model import DirectPredictionModule
from omegaconf import DictConfig
from PIL import Image

# Constants
IMAGE_PATH = "../CM_datasets/images/DAB_AN/620.png"
CONF_MAP_PATH = "../CM_datasets/ultranerf/DAB_AN/620.png"
CHECKPOINT_PATH = "/home/vanessa_share/bugra/dataset_project/e2e_training/logs/DirectPredictionModule/version_2/checkpoints/epoch=23-step=63552.ckpt"
MODEL_NAME = "DirectPredictionModule"
OUTPUT_DIR = "output"


@hydra.main(config_path="configs", config_name="config", version_base=None)
def generate(cfg: DictConfig) -> None:
    """
    Load the pre-trained model, process the test image and confidence map,
    generate predictions, visualize the difference, and save a single matplotlib plot.
    """

    # Create Experiment Directory
    experiment_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load the model from checkpoint
    model: pl.LightningModule = DirectPredictionModule.load_from_checkpoint(
        CHECKPOINT_PATH
    )

    # Load the dataset and apply transformations
    dataset = CMDataset(
        images_dir=cfg.data.images_dir,
        confidence_maps_dir=cfg.data.confidence_maps_dir,
        split="test",
    )

    # Load and process the test image and confidence map
    test_image_raw = Image.open(IMAGE_PATH).convert("L")
    test_conf_map = Image.open(CONF_MAP_PATH).convert("L")

    # Apply dataset transformations
    test_image, _ = dataset.transform(test_image_raw, test_conf_map)
    test_image = test_image.unsqueeze(0).cuda()

    # Set model to evaluation mode and generate prediction
    model.eval()
    with torch.no_grad():
        pred = model(test_image)

    # Post-process prediction
    pred = torch.clamp(pred, 0, 1).squeeze(0).squeeze(0)
    pred_img = Image.fromarray((pred * 255).cpu().numpy().astype(np.uint8))
    pred_img = pred_img.resize(test_image_raw.size, resample=Image.BILINEAR)

    # Convert images to numpy arrays and normalize to 0-1
    pred_np = np.array(pred_img, dtype=np.float32) / 255.0
    image_np = np.array(test_image_raw, dtype=np.float32) / 255.0
    conf_np = np.array(test_conf_map, dtype=np.float32) / 255.0
    diff_np = np.abs(conf_np - pred_np)

    # Save the prediction and ground truth images
    patient_name = os.path.basename(
        os.path.dirname(IMAGE_PATH)
    )  # Extract patient name from path
    slice_num = os.path.splitext(os.path.basename(IMAGE_PATH))[
        0
    ]  # Extract slice number from path

    save_dir = os.path.join(experiment_dir, patient_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save path
    save_path = os.path.join(save_dir, f"{slice_num}.png")

    # Create and save a matplotlib plot with all images
    _plot_all_images(image_np, pred_np, conf_np, diff_np, save_path)


def _plot_all_images(
    image_np: np.ndarray,
    pred_np: np.ndarray,
    conf_np: np.ndarray,
    diff_np: np.ndarray,
    save_path: str,
) -> None:
    """
    Create a matplotlib plot with subplots for the original image, prediction, ground truth confidence map, and difference image.
    All images are normalized to 0-1, with appropriate colorbars.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    # Plot original image
    im0 = axes[0].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original Image")
    fig.colorbar(im0, ax=axes[0])

    # Plot predicted confidence map
    im1 = axes[1].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicted Confidence Map")
    fig.colorbar(im1, ax=axes[1])

    # Plot ground truth confidence map
    im2 = axes[2].imshow(conf_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth Confidence Map")
    fig.colorbar(im2, ax=axes[2])

    # Plot difference image
    im3 = axes[3].imshow(diff_np, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Difference (GT - Pred)")
    fig.colorbar(im3, ax=axes[3])

    # Tidy up the layout
    plt.tight_layout()

    # Save the plot to the same directory as the original image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    generate()
