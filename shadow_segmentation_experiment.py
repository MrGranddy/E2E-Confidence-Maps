import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from random_forest_segmentor import Segmentor
from utils import (
    convert_labels_to_shadow_map,
    depatchify_single_image_prediction,
    generate_patch_dataset,
    patchify_single_image,
    read_labels,
    split_views,
)

SEED = 42
NUM_PATCHES_EACH_SPLIT = 10000
PATCH_SIZE = 5

NUM_SAMPLE_IMAGES_TO_SAVE = 5
OUTPUT_PATH = "shadow_segmentation_experiment_results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

np.random.seed(SEED)


def run_experiment(
    train_image_splits: List[np.ndarray],
    train_shadow_map_splits: List[np.ndarray],
    train_conf_map_splits: Optional[List[np.ndarray]],
    patch_idxs: List[List[tuple]],
    test_images: np.ndarray,
):

    train_patches, train_labels = generate_patch_dataset(
        {
            "image": train_image_splits,
            "shadow_map": train_shadow_map_splits,
            "conf_map": train_conf_map_splits,
        },
        PATCH_SIZE,
        patch_idxs,
    )

    segmentor = Segmentor(seed=SEED)
    segmentor.train(train_patches, train_labels)

    predictions = np.zeros(
        (
            test_images.shape[0],
            test_images.shape[1] - PATCH_SIZE + 1,
            test_images.shape[2] - PATCH_SIZE + 1,
        ),
        dtype=np.uint8,
    )
    for i, test_image in enumerate(test_images):
        test_image_patches = patchify_single_image(test_image, PATCH_SIZE)
        test_image_predictions = segmentor.predict(test_image_patches)
        test_shadow_map_predictions = depatchify_single_image_prediction(
            test_image_predictions, test_image.shape, PATCH_SIZE
        )

        predictions[i] = test_shadow_map_predictions

    return predictions


if __name__ == "__main__":

    spine_images_path = "G:/masters-thesis/final_data/original/spine/images.npy"
    liver_images_path = "G:/masters-thesis/final_data/original/liver/images.npy"

    spine_split_idxs = [
        0,
        159,
        313,
        465,
        618,
        769,
        921,
        1067,
        1217,
        1370,
        1534,
        1682,
        1823,
        1975,
        2123,
        2253,
        2389,
    ]
    liver_split_idxs = [0, 200, 400, 600, 800, 1000, 1200]

    spine_labels_path = "G:/masters-thesis/data/spine_labels/all.nii"
    liver_labels_path = "G:/masters-thesis/data/liver_labels/mask.npy"

    methods_old = ["acyclic", "karamalis", "mid", "min", "ultranerf"]
    methods_new = ["reg0", "reg01", "reg001", "reg0001"]

    cm_paths_spine = {
        method: f"G:/masters-thesis/final_data/{method}/spine/full/conf_map.npy"
        for method in methods_old
    }

    for method in methods_new:
        cm_paths_spine[method] = (
            f"G:/masters-thesis/consistency_experiment_data/spine/full/cms_{method}.npy"
        )

    labels_spine = read_labels(spine_labels_path)
    shadow_map_spine = convert_labels_to_shadow_map(labels_spine, 1)

    cm_paths_liver = {
        method: f"G:/masters-thesis/final_data/{method}/liver/full/conf_map.npy"
        for method in methods_old
    }

    for method in methods_new:
        cm_paths_liver[method] = (
            f"G:/masters-thesis/consistency_experiment_data/liver/full/cms_{method}.npy"
        )

    labels_liver = read_labels(liver_labels_path)
    shadow_map_liver = convert_labels_to_shadow_map(labels_liver, 1)

    # Load images
    spine_images = np.load(spine_images_path)
    liver_images = np.load(liver_images_path)

    # Split images
    spine_splits = split_views(spine_images, spine_split_idxs)
    liver_splits = split_views(liver_images, liver_split_idxs)

    # Split shadow maps
    shadow_map_spine_splits = split_views(shadow_map_spine, spine_split_idxs)
    shadow_map_liver_splits = split_views(shadow_map_liver, liver_split_idxs)

    # Generate random idxs for patches
    spine_patch_idxs = []
    liver_patch_idxs = []

    for split in spine_splits:
        spine_patch_idxs.append([])

        for _ in range(NUM_PATCHES_EACH_SPLIT):
            d = np.random.randint(0, split.shape[0])
            x = np.random.randint(0, split.shape[1] - PATCH_SIZE)
            y = np.random.randint(0, split.shape[2] - PATCH_SIZE)
            spine_patch_idxs[-1].append((d, x, y))

    for split in liver_splits:
        liver_patch_idxs.append([])
        for _ in range(NUM_PATCHES_EACH_SPLIT):
            d = np.random.randint(0, split.shape[0])
            x = np.random.randint(0, split.shape[1] - PATCH_SIZE)
            y = np.random.randint(0, split.shape[2] - PATCH_SIZE)
            liver_patch_idxs[-1].append((d, x, y))

    # Run experiments

    print("No Confidence Map - Spine")
    for idx, split in enumerate(spine_splits):
        test_images = split
        test_shadow_map = shadow_map_spine_splits[idx]

        train_image_splits = spine_splits[:idx] + spine_splits[idx + 1 :]
        train_shadow_map_splits = (
            shadow_map_spine_splits[:idx] + shadow_map_spine_splits[idx + 1 :]
        )
        train_conf_map_splits = None
        train_patch_idxs = spine_patch_idxs[:idx] + spine_patch_idxs[idx + 1 :]

        predictions = run_experiment(
            train_image_splits,
            train_shadow_map_splits,
            train_conf_map_splits,
            train_patch_idxs,
            test_images,
        )

        trimmed_ground_truth = test_shadow_map[
            :, PATCH_SIZE // 2 : -(PATCH_SIZE // 2), PATCH_SIZE // 2 : -(PATCH_SIZE // 2)
        ]

        print(f"Spine - No Confidence Map - Split {idx}")
        print(f"Accuracy: {np.mean(predictions == trimmed_ground_truth)}")
        print(
            f"Pred Shape: {predictions.shape}, GT Shape: {trimmed_ground_truth.shape}"
        )

        os.makedirs(os.path.join(OUTPUT_PATH, "spine_no_cm"), exist_ok=True)
        for i, prediction in enumerate(predictions[:NUM_SAMPLE_IMAGES_TO_SAVE]):
            plt.imsave(
                os.path.join(
                    OUTPUT_PATH,
                    "spine_no_cm",
                    f"split_{idx}_prediction_{i}.png",
                ),
                prediction,
                cmap="gray",
            )
