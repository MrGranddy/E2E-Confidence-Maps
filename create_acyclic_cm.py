import argparse
import os

import numpy as np
from confidence_with_dg import confidence_map
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="render",
        help="Path to the directory containing the rendered images",
    )
    parser.add_argument(
        "--participant_name", type=str, default="test", help="Name of the participant"
    )
    parser.add_argument(
        "--destination_dir",
        type=str,
        default="output",
        help="Path to the directory where the confidence maps will be saved",
    )
    args = parser.parse_args()

    # Set saving parameters
    confidence_map_type = "acyclic"

    # Get parameters
    image_paths = [
        os.path.join(args.input_path, image_name)
        for image_name in os.listdir(args.input_path)
    ]
    num_images = len(image_paths)
    image_shape = np.array(Image.open(image_paths[0]).convert("L")).shape

    # Create confidence map array
    confidence_maps = np.zeros((num_images, *image_shape), dtype=np.float32)

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("L")
        image_array = np.array(image).astype(np.float32) / 255.0

        confidence_maps[i] = confidence_map(image_array)

        if (i + 1) % 20 == 0:
            print(
                f"Processed {i + 1} of {num_images} images for participant {args.participant_name}"
            )

    if not os.path.exists(args.destination_dir):
        os.makedirs(args.destination_dir)

    parent_path = os.path.join(args.destination_dir, confidence_map_type)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    save_path = os.path.join(parent_path, f"{args.participant_name}.npy")

    # Save the confidence maps
    np.save(save_path, confidence_maps)
