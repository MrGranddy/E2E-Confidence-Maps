import argparse
import os
import shutil

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="render", help="Path to the directory containing the rendered images")
    parser.add_argument("--participant_name", type=str, default="test", help="Name of the participant")
    parser.add_argument("--destination_dir", type=str, default="output", help="Path to the directory where the confidence maps will be saved")
    args = parser.parse_args()

    # Set saving parameters
    os.makedirs(os.path.join(args.destination_dir), exist_ok=True)

    # Set participant path
    participant_path = os.path.join(args.destination_dir, args.participant_name)

    # Copy the parameter files
    if os.path.isdir(participant_path):
        shutil.rmtree(participant_path)
    shutil.copytree(args.input_path, participant_path)
