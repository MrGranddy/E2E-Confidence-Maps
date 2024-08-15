import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from random_forest_segmentor import Segmentor
from utils import (
    convert_labels_to_shadow_map,
    depatchify_single_image_prediction,
    generate_patch_dataset,
    patchify_single_image,
    read_labels,
    split_views,
    evaluate_segmentation,
)

SEED = 42
NUM_PATCHES_EACH_SPLIT = 10000
PATCH_SIZE = 5

NUM_TEST_IMAGES_TO_EVALUATE = 10

OUTPUT_PATH = "shadow_segmentation_experiment_results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

np.random.seed(SEED)

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
spine_images = np.load(spine_images_path).astype(np.float32) / 255.0
liver_images = np.load(liver_images_path).astype(np.float32) / 255.0

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

# Sample test image idxs for each split of each dataset
spine_test_image_idxs = []
liver_test_image_idxs = []

for split in spine_splits:
    spine_test_image_idxs.append(
        np.random.choice(np.arange(split.shape[0]), NUM_TEST_IMAGES_TO_EVALUATE)
    )

for split in liver_splits:
    liver_test_image_idxs.append(
        np.random.choice(np.arange(split.shape[0]), NUM_TEST_IMAGES_TO_EVALUATE)
    )


def run_single_split(
    train_image_splits: List[np.ndarray],
    train_shadow_map_splits: List[np.ndarray],
    train_conf_map_splits: Optional[List[np.ndarray]],
    patch_idxs: List[List[tuple]],
    test_images: np.ndarray,
    test_conf_maps: Optional[np.ndarray],
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

    n_estimators = 50 if train_conf_map_splits is not None else 25

    segmentor = Segmentor(n_estimators=n_estimators, seed=SEED)
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

        if test_conf_maps is not None:
            test_conf_map = test_conf_maps[i]
        else:
            test_conf_map = None

        test_image_patches = patchify_single_image(
            test_image, test_conf_map, PATCH_SIZE
        )
        test_image_predictions = segmentor.predict(test_image_patches)
        test_shadow_maps_predictions = depatchify_single_image_prediction(
            test_image_predictions, test_image.shape, PATCH_SIZE
        )

        predictions[i] = test_shadow_maps_predictions

    return predictions


def run_experiment(config):

    dataset = config["dataset"]
    method = config["method"]
    splits = config["splits"]
    shadow_map_splits = config["shadow_map_splits"]
    conf_map_path = config["conf_map_path"]
    split_idxs = config["split_idxs"]
    patch_idxs = config["patch_idxs"]
    test_image_idxs = config["test_image_idxs"]

    if method == "no_cm":
        conf_map_splits = None
    else:
        conf_map = np.load(conf_map_path).astype(np.float32) / 255.0
        conf_map_splits = split_views(conf_map, split_idxs)

    test_metrics = {
        "precision": [],
        "dice": [],
        "hausdorff": [],
    }

    print(f"Running experiment for {dataset} - {method}")

    for idx, split in enumerate(splits):
        test_images = split[test_image_idxs[idx]]
        test_shadow_maps = shadow_map_splits[idx][test_image_idxs[idx]]
        test_conf_maps = (
            conf_map_splits[idx][test_image_idxs[idx]]
            if conf_map_splits is not None
            else None
        )

        train_image_splits = splits[:idx] + splits[idx + 1 :]
        train_shadow_map_splits = shadow_map_splits[:idx] + shadow_map_splits[idx + 1 :]

        if conf_map_splits is not None:
            train_conf_map_splits = conf_map_splits[:idx] + conf_map_splits[idx + 1 :]
        else:
            train_conf_map_splits = None

        train_patch_idxs = patch_idxs[:idx] + patch_idxs[idx + 1 :]

        predictions = run_single_split(
            train_image_splits,
            train_shadow_map_splits,
            train_conf_map_splits,
            train_patch_idxs,
            test_images,
            test_conf_maps,
        )

        trimmed_ground_truth = test_shadow_maps[
            :,
            PATCH_SIZE // 2 : -(PATCH_SIZE // 2),
            PATCH_SIZE // 2 : -(PATCH_SIZE // 2),
        ]

        run_metrics = evaluate_segmentation(predictions, trimmed_ground_truth)

        test_metrics["precision"].append(run_metrics["precision"])
        test_metrics["dice"].append(run_metrics["dice"])
        test_metrics["hausdorff"].append(run_metrics["hausdorff"])

    test_metrics["precision"] = np.array(test_metrics["precision"])
    test_metrics["dice"] = np.array(test_metrics["dice"])
    test_metrics["hausdorff"] = np.array(test_metrics["hausdorff"])

    print(f"Precision: {test_metrics['precision'].mean()}")
    print(f"Dice: {test_metrics['dice'].mean()}")
    print(f"Hausdorff: {test_metrics['hausdorff'].mean()}")

    experiment_output_dir = os.path.join(OUTPUT_PATH, f"{dataset}-{method}")
    os.makedirs(experiment_output_dir, exist_ok=True)

    for i in range(test_images.shape[0]):

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(test_images[i], cmap="gray")
        ax[0].set_title("Image")

        ax[1].imshow(predictions[i], cmap="gray")
        ax[1].set_title("Prediction")

        ax[2].imshow(trimmed_ground_truth[i], cmap="gray")
        ax[2].set_title("Ground Truth")

        plt.tight_layout()

        plt.savefig(
            os.path.join(experiment_output_dir, f"plt_{i}.png"), bbox_inches="tight"
        )

        plt.close()

        Image.fromarray((test_images[i] * 255).astype(np.uint8)).save(
            os.path.join(experiment_output_dir, f"image_{i}.png")
        )

        Image.fromarray((predictions[i] * 255).astype(np.uint8)).save(
            os.path.join(experiment_output_dir, f"pred_{i}.png")
        )

        Image.fromarray((trimmed_ground_truth[i] * 255).astype(np.uint8)).save(
            os.path.join(experiment_output_dir, f"gt_{i}.png")
        )


    


if __name__ == "__main__":
    configs = []

    datasets = ["liver"]
    methods = [
        "acyclic",
        "karamalis",
        "mid",
        "min",
        "ultranerf",
        "reg0",
        "reg01",
        "reg001",
        "reg0001",
        "no_cm",
    ]

    for dataset in datasets:
        for method in methods:
            if dataset == "spine":
                image_splits = spine_splits
                shadow_map_splits = shadow_map_spine_splits
                patch_idxs = spine_patch_idxs
                cm_paths = cm_paths_spine
                split_idxs = spine_split_idxs
                test_image_idxs = spine_test_image_idxs
            else:
                image_splits = liver_splits
                shadow_map_splits = shadow_map_liver_splits
                patch_idxs = liver_patch_idxs
                cm_paths = cm_paths_liver
                split_idxs = liver_split_idxs
                test_image_idxs = liver_test_image_idxs

            conf_map_path = cm_paths[method] if method != "no_cm" else None

            configs.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "splits": image_splits,
                    "shadow_map_splits": shadow_map_splits,
                    "conf_map_path": conf_map_path,
                    "patch_idxs": patch_idxs,
                    "split_idxs": split_idxs,
                    "test_image_idxs": test_image_idxs,
                }
            )

    for config in configs:
        run_experiment(config)
