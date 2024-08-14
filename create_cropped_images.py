import argparse
import os

import numpy as np

data_main_path = "G:/masters-thesis/final_data"
output_path = "G:/masters-thesis/consistency_experiment_data"

data_names = ["leg", "liver", "spine"]
modes = ["full", "left", "right", "partial"]

os.makedirs(output_path, exist_ok=True)

for image_name in data_names:

    image = np.load(
        os.path.join(data_main_path, "original", f"{image_name}", "images.npy")
    )

    N, H, W = image.shape

    print(f"Processing {image_name}...")
    images_path = os.path.join(output_path, image_name)
    os.makedirs(images_path, exist_ok=True)

    for mode in modes:

        print(f"Processing {mode}...")
        mode_path = os.path.join(images_path, mode)
        os.makedirs(mode_path, exist_ok=True)

        if mode == "full":
            image_mode = image
        elif mode == "left":
            crop = W // 3
            image_mode = image[:, :, :-crop]
        elif mode == "right":
            crop = W // 3
            image_mode = image[:, :, crop:]
        elif mode == "partial":
            crop = H // 3
            image_mode = image[:, :-crop, :]

        np.save(os.path.join(mode_path, "images.npy"), image_mode)
