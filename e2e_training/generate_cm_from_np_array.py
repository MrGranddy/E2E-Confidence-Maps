import argparse
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from e2e_training.dataset import CMDataset
from e2e_training.model import DirectPredictionModule

BATCH_SIZE = 128
TARGET_WIDTH = 128
TARGET_HEIGHT = 256
MEAN = 0.5
STD = 0.5


def process_batch(batch: np.ndarray) -> np.ndarray:

    processed_batch = np.zeros(
        (batch.shape[0], TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32
    )

    for i, img in enumerate(batch):
        img = Image.fromarray(img)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
        processed_batch[i] = np.array(img).astype(np.float32)

    processed_batch = processed_batch / 255.0
    processed_batch = (processed_batch - MEAN) / STD

    return processed_batch


def create_cms(array: np.ndarray, model: pl.LightningModule) -> np.ndarray:

    n, h, w = array.shape
    num_batches = n // BATCH_SIZE
    if n % BATCH_SIZE != 0:
        num_batches += 1

    # Create an empty array to store the confidence maps
    resized_cms = np.zeros((n, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)

    # Iterate over the array and create the confidence maps as batched tensors
    for i in range(num_batches):

        curr_batch_size = min(BATCH_SIZE, n - i * BATCH_SIZE)

        # Extract the current batch
        batch = array[i * BATCH_SIZE : i * BATCH_SIZE + curr_batch_size]

        # Normalize the batch and convert to tensor
        batch = process_batch(batch)
        batch = torch.from_numpy(batch).unsqueeze(1).float().cuda()

        with torch.no_grad():
            cm_batch = model(batch)

        cm_batch = torch.clamp(cm_batch, 0, 1).squeeze(1).cpu().numpy() * 255

        resized_cms[i * BATCH_SIZE : i * BATCH_SIZE + curr_batch_size] = cm_batch

    cms = np.zeros((n, h, w), dtype=np.uint8)

    for i in range(n):
        img = Image.fromarray(resized_cms[i])
        img = img.resize((w, h), Image.BILINEAR)
        cms[i] = np.array(img)

    return cms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate confidence maps from a numpy array"
    )
    parser.add_argument("input", type=str, help="Path to the input numpy array")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("output", type=str, help="Path to the output directory")
    args = parser.parse_args()

    # Load the numpy array
    array = np.load(args.input)

    # Load the model from checkpoint
    model = DirectPredictionModule.load_from_checkpoint(args.checkpoint)
    model = model.cuda()
    model.eval()

    # Generate confidence maps
    cms = create_cms(array, model)

    # Save the confidence maps
    np.save(args.output, cms)
