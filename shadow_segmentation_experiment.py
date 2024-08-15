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
NUM_PATCHES_EACH_SPLIT = 10
PATCH_SIZE = 5

np.random.seed(SEED)

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
    shadow_map_spine = split_views(shadow_map_spine, spine_split_idxs)
    shadow_map_liver = split_views(shadow_map_liver, liver_split_idxs)

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

    # Conduct experiment without confidence maps
    print("Experiment without confidence maps")
    print("Spine:")

    patches, labels = generate_patch_dataset(
        {"image": spine_splits, "shadow_map": shadow_map_spine},
        PATCH_SIZE,
        spine_patch_idxs,
    )
    segmentor = Segmentor(seed=SEED)
    segmentor.train(patches, labels)

    print(f"Patches shape: {patches.shape}, Labels shape: {labels.shape}")

    print("Liver:")

    patches, labels = generate_patch_dataset(
        {"image": liver_splits, "shadow_map": shadow_map_liver},
        PATCH_SIZE,
        liver_patch_idxs,
    )

    print(f"Patches shape: {patches.shape}, Labels shape: {labels.shape}")
