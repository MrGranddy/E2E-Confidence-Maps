# This script is to be used after creating confidence maps of any kind
# It matches input images and confidence maps in numpy arrays to make them into a
# batch-wise readable

# Datasets dir shoudld contain datasets with {DATASET_NAME} dirs of ultranerf format
# Confidence maps dir should contain confidence maps with {DATASET_NAME}.npy, the confidence maps are in shape (N, H, W)
# Where image idx in ultra nerf dataset is the same as the confidence map idx in the confidence map array

import argparse
import os

import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image-cm pairs from datasets")

    parser.add_argument(
        "--datasets_dir", type=str, required=True, help="Directory of the datasets"
    )
    parser.add_argument(
        "--confidence_maps_dir",
        type=str,
        required=True,
        help="Directory of the confidence maps",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory of the output"
    )
    parser.add_argument(
        "--cm_type", type=str, required=True, help="Type of the confidence map"
    )

    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    confidence_maps_dir = args.confidence_maps_dir
    output_dir = args.output_dir
    cm_type = args.cm_type

    dataset_names = os.listdir(datasets_dir)

    os.makedirs(output_dir, exist_ok=True)

    output_images_dir = os.path.join(output_dir, "images")
    output_cm_dir = os.path.join(output_dir, cm_type)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_cm_dir, exist_ok=True)

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(datasets_dir, dataset_name)
        confidence_map_path = os.path.join(confidence_maps_dir, dataset_name + ".npy")

        dataset_images_dir = os.path.join(output_images_dir, dataset_name)
        dataset_confidence_maps_dir = os.path.join(output_cm_dir, dataset_name)

        os.makedirs(dataset_images_dir, exist_ok=True)
        os.makedirs(dataset_confidence_maps_dir, exist_ok=True)

        images_path = os.path.join(dataset_dir, "images.npy")

        try:
            images = np.load(images_path)
        except FileNotFoundError:
            print(f"{dataset_name} images not found")
            continue
        try:
            confidence_maps = np.load(confidence_map_path)
        except FileNotFoundError:
            print(f"{dataset_name} confidence maps not found")
            continue

        if images.shape[0] != confidence_maps.shape[0]:
            print(
                f"{dataset_name} image shape {images.shape} and confidence map shape {confidence_maps.shape} mismatch"
            )
            continue

        # Check if format is 0-255 or 0-1, if 0-1 convert to 0-255 uint8
        if images.max() <= 1:
            images = (images * 255).astype(np.uint8)

        if confidence_maps.max() <= 1:
            confidence_maps = (confidence_maps * 255).astype(np.uint8)

        for i in range(images.shape[0]):
            image = Image.fromarray(images[i])
            confidence_map = Image.fromarray(confidence_maps[i])

            image.save(os.path.join(dataset_images_dir, f"{i}.png"))
            confidence_map.save(os.path.join(dataset_confidence_maps_dir, f"{i}.png"))

        print(f"{dataset_name} done")

    print("All done")
