import matplotlib.pyplot as plt
import numpy as np

from utils import convert_labels_to_shadow_map, read_labels

spine_labels_path = "G:/masters-thesis/data/spine_labels/all.nii"
liver_labels_path = "G:/masters-thesis/data/liver_labels/mask.npy"

methods = ["acyclic", "karamalis", "mid", "min", "ultranerf"]

cm_paths = {
    method: f"G:/masters-thesis/final_data/{method}/spine/full/conf_map.npy"
    for method in methods
}

methods = ["reg0", "reg01", "reg001", "reg0001"]

for method in methods:
    cm_paths[method] = (
        f"G:/masters-thesis/consistency_experiment_data/spine/full/cms_{method}.npy"
    )

labels = read_labels(spine_labels_path)
shadow_map = convert_labels_to_shadow_map(labels, 1)

for method, path in cm_paths.items():

    cms = np.load(path).astype(np.float32) / 255.0

    shadow_confidence = cms[shadow_map]
    shadow_confidence_mean = shadow_confidence.mean()
    shadow_confidence_std = shadow_confidence.std()

    print(
        f"Method: {method}, Dataset spine, Confidence: {shadow_confidence_mean:.4f} ± {shadow_confidence_std:.4f}"
    )

methods = ["acyclic", "karamalis", "mid", "min", "ultranerf"]

cm_paths = {
    method: f"G:/masters-thesis/final_data/{method}/liver/full/conf_map.npy"
    for method in methods
}

methods = ["reg0", "reg01", "reg001", "reg0001"]

for method in methods:
    cm_paths[method] = (
        f"G:/masters-thesis/consistency_experiment_data/liver/full/cms_{method}.npy"
    )

labels = read_labels(liver_labels_path)
shadow_map = convert_labels_to_shadow_map(labels, 1)

for method, path in cm_paths.items():

    cms = np.load(path).astype(np.float32) / 255.0

    shadow_confidence = cms[shadow_map]
    shadow_confidence_mean = shadow_confidence.mean()
    shadow_confidence_std = shadow_confidence.std()

    print(
        f"Method: {method}, Dataset liver, Confidence: {shadow_confidence_mean:.4f} ± {shadow_confidence_std:.4f}"
    )
