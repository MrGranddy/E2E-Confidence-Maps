from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def visualize_predictions(
    images: torch.Tensor,
    conf_maps: torch.Tensor,
    preds: torch.Tensor,
    num_images_to_log: int = 4,
) -> List[torch.Tensor]:

    img_tensors = []

    for i in range(num_images_to_log):
        # Convert tensors to numpy arrays for plotting
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        gt = conf_maps[i].detach().cpu().numpy().transpose(1, 2, 0)
        pred = preds[i].detach().cpu().numpy().transpose(1, 2, 0)

        # Create a matplotlib figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original image
        im0 = axs[0].imshow(img, cmap="gray")
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Plot the ground truth
        im1 = axs[1].imshow(gt, cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")
        cbar1 = fig.colorbar(
            im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04
        )

        # Plot the prediction
        im2 = axs[2].imshow(pred, cmap="gray")
        axs[2].set_title("Prediction")
        axs[2].axis("off")
        cbar2 = fig.colorbar(
            im2, ax=axs[2], orientation="vertical", fraction=0.046, pad=0.04
        )

        # Adjust layout to be tight and space-effective
        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        # Convert buffer to an image tensor
        img_tensor = torch.tensor(
            np.array(Image.open(buf)).transpose(2, 0, 1) / 255.0, dtype=torch.float32
        )
        img_tensors.append(img_tensor)

    return img_tensors
