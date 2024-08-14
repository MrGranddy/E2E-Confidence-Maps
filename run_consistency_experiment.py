import os

import numpy as np

cm_types = ["acyclic", "karamalis", "mid", "min", "ultranerf"]
datasets = ["leg", "liver", "spine"]
crops = ["full", "left", "right", "partial"]

data_path = "G:/masters-thesis/final_data"

path_tree = {
    cm_type: {
        dset: {
            crop: os.path.join(data_path, cm_type, dset, crop, "conf_map.npy")
            for crop in crops
        }
        for dset in datasets
    }
    for cm_type in cm_types
}

data_path = "G:/masters-thesis/consistency_experiment_data"

cm_types = ["reg0", "reg01", "reg001", "reg0001"]
datasets = ["leg", "liver", "spine"]
crops = ["full", "left", "right", "partial"]

for cm_type in cm_types:

    path_tree[cm_type] = {}

    for dataset in datasets:

        path_tree[cm_type][dataset] = {}

        for crop in crops:

            path_tree[cm_type][dataset][crop] = os.path.join(data_path, dataset, crop, f"cms_{cm_type}.npy")



def calculate_psnr(img1, img2):

    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)

    # Avoid division by zero
    if mse == 0:
        mse = 1e-10

    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


for cm_type in path_tree:
    for dataset in path_tree[cm_type]:

        # Load full
        full = np.load(path_tree[cm_type][dataset]["full"]).astype(np.float32) / 255.0

        # First left and right experiments
        left = np.load(path_tree[cm_type][dataset]["left"]).astype(np.float32) / 255.0
        right = np.load(path_tree[cm_type][dataset]["right"]).astype(np.float32) / 255.0

        left_start = full.shape[2] - right.shape[2]
        right_end = 2 * left.shape[2] - full.shape[2]

        left_intersect = left[:, :, left_start:]
        right_intersect = right[:, :, :right_end]

        # Calculate PSNR for each slice and add to list
        psnrs = []
        for i in range(full.shape[0]):
            psnrs.append(calculate_psnr(left_intersect[i], right_intersect[i]))

        # Calculate mean and std
        mean_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)

        print(
            f"PSNR for {cm_type}-{dataset} left-right intersection: {mean_psnr:.2f} ± {std_psnr:.2f}"
        )


for cm_type in path_tree:
    for dataset in path_tree[cm_type]:

        # Load full
        full = np.load(path_tree[cm_type][dataset]["full"]).astype(np.float32) / 255.0

        # First left and right experiments
        try:
            partial = np.load(path_tree[cm_type][dataset]["partial"]).astype(np.float32) / 255.0
        except FileNotFoundError:
            print(f"Partial not found for {cm_type}-{dataset}")
            continue

        full_intersect = full[:, partial.shape[1] - 5 : partial.shape[1], :]
        partial_intersect = partial[:, -5:, :]

        # Calculate PSNR for each slice and add to list
        psnrs = []
        for i in range(full_intersect.shape[0]):
            psnrs.append(calculate_psnr(full_intersect[i], partial_intersect[i]))

        # Calculate mean and std
        mean_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)

        print(
            f"PSNR for {cm_type}-{dataset} full-partial intersection: {mean_psnr:.2f} ± {std_psnr:.2f}"
        )
