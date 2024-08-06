import argparse
import os

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="render", help="Path to the directory containing the rendered images")
    parser.add_argument("--participant_name", type=str, default="test", help="Name of the participant")
    parser.add_argument("--destination_dir", type=str, default="output", help="Path to the directory where the confidence maps will be saved")
    args = parser.parse_args()

    # Set saving parameters
    confidence_map_type = "ultranerf"

    # Load the rendered images
    confidence_path = os.path.join(args.input_path, "confidence_maps.npy")
    confidence_maps = np.load(confidence_path)

    if not os.path.exists(args.destination_dir):
        os.makedirs(args.destination_dir)

    parent_path = os.path.join(args.destination_dir, confidence_map_type)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    save_path = os.path.join(parent_path, f"{args.participant_name}.npy")

    # Save the confidence maps
    np.save(save_path, confidence_maps)