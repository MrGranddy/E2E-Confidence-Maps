import argparse
import os

import numpy as np

from e2e_training.generate_cm_from_np_array import create_cms
from e2e_training.model import DirectPredictionModule

data_path = "G:/masters-thesis/consistency_experiment_data"

checkpoins = {
    "reg0": "G:/E2E-Confidence-Maps/e2e_training/weights/direct_prediction_reg0.ckpt",
    "reg01": "G:/E2E-Confidence-Maps/e2e_training/weights/direct_prediction_reg01.ckpt",
    "reg001": "G:/E2E-Confidence-Maps/e2e_training/weights/direct_prediction_reg001.ckpt",
    "reg0001": "G:/E2E-Confidence-Maps/e2e_training/weights/direct_prediction_reg0001.ckpt",
}

data_names = ["leg", "liver", "spine"]
modes = ["full", "left", "right", "partial"]

for reg, checkpoint in checkpoins.items():

    print(f"Processing {reg}...")
    model = DirectPredictionModule.load_from_checkpoint(checkpoint)
    model = model.cuda()
    model.eval()

    for image_name in data_names:

        print(f"Processing {image_name}...")
        images_path = os.path.join(data_path, image_name)

        for mode in modes:
            print(f"Processing {mode}...")
            mode_path = os.path.join(images_path, mode)
            array_path = os.path.join(mode_path, "images.npy")

            array = np.load(array_path)
            cms = create_cms(array, model)

            cm_path = os.path.join(mode_path, f"cms_{reg}.npy")
            np.save(cm_path, cms)

print("Done!")
