from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from scipy.spatial.distance import directed_hausdorff


def convert_labels_to_shadow_map(labels: np.ndarray, bone_idx: int) -> np.ndarray:

    bone_mask = labels == bone_idx

    shadow_mask = np.zeros_like(bone_mask)

    for i in range(bone_mask.shape[0]):
        for j in range(bone_mask.shape[2]):
            line = bone_mask[i, :, j]
            if np.any(line):
                bottom = np.where(line)[0][-1]
                shadow_mask[i, bottom:, j] = 1

    return shadow_mask


def read_images(path: str) -> np.ndarray:

    image = np.load(path)
    image = image.astype(np.float32) / 255.0

    return image


def read_labels(path: str) -> np.ndarray:

    if path.endswith(".npy"):
        label_images = np.load(path)

    else:
        label_images = nib.load(path).get_fdata()[..., 0, :]
        label_images = np.transpose(label_images, (2, 1, 0))

    return label_images.astype(np.uint8)


def split_views(image: np.ndarray, view_idxs: List[int]) -> List[np.ndarray]:

    splits = []
    for i in range(len(view_idxs) - 1):
        start = view_idxs[i]
        end = view_idxs[i + 1]
        splits.append(image[start:end, ...])

    return splits


def generate_patch_dataset(
    dataset: Dict[str, List[np.ndarray]],
    patch_size: int,
    patch_idxs: List[List[Tuple[int, int, int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates patches from the dataset.

    Args:
        dataset (Dict[str, List[np.ndarray]]): A dictionary containing the dataset. Where the keys are 'image', 'conf_map' and 'shadow_map'. Each value is a list of numpy arrays where each element is a split of the full US image.
        patch_size (int): The size of the patches.
        patch_idxs (List[List[Tuple[int, int, int]]]): A list of lists where each inner list contains the indices of the patches to generate for each split.

    Returns:
        patches (np.ndarray): The flattened patches of shape N x D where
            N = num_patches_each_split * num_splits and D = patch_size^2 or patch_size^2 * 2 if conf_map is not None.
        labels (np.ndarray): The labels of the patches. (The value of the shadow map at the center of the patch).
    """

    image_splits = dataset["image"]
    conf_map_splits = dataset.get("conf_map", None)
    shadow_map_splits = dataset["shadow_map"]

    patches = []
    labels = []

    for i in range(len(image_splits)):
        image = image_splits[i]
        conf_map = conf_map_splits[i] if conf_map_splits is not None else None
        shadow_map = shadow_map_splits[i]

        split_patch_idxs = patch_idxs[i]

        for d, x, y in split_patch_idxs:

            patch_image = image[d, x : x + patch_size, y : y + patch_size]
            patch_label = shadow_map[d, x + patch_size // 2, y + patch_size // 2]

            if conf_map is not None:
                patch_conf_map = conf_map[d, x : x + patch_size, y : y + patch_size]
                patch = np.concatenate(
                    [patch_image.flatten(), patch_conf_map.flatten()]
                )
            else:
                patch = patch_image.flatten()

            patches.append(patch)
            labels.append(patch_label)

    patches = np.stack(patches)
    labels = np.array(labels)

    return patches, labels


def patchify_single_image(
    image: np.ndarray, conf_map: Optional[np.ndarray], patch_size: int
) -> np.ndarray:
    """Generates patches from a single image.

    Args:
        image (np.ndarray): The image to generate patches from.
        patch_size (int): The size of the patches.

    Returns:
        patches (np.ndarray): The flattened patches of shape N x D where
            N = num_patches and D = patch_size^2.
    """

    patches = []

    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):

            image_vector = image[i : i + patch_size, j : j + patch_size].flatten()

            if conf_map is not None:
                conf_map_vector = conf_map[
                    i : i + patch_size, j : j + patch_size
                ].flatten()
                patch = np.concatenate([image_vector, conf_map_vector])
            else:
                patch = image_vector

            patches.append(patch)

    patches = np.stack(patches)

    return patches


def depatchify_single_image_prediction(
    predictions: np.ndarray, image_shape: Tuple[int, int], patch_size: int
) -> np.ndarray:
    """Depatchifies the predictions of a single image.

    Args:
        predictions (np.ndarray): The predictions of the patches.
        image_shape (Tuple[int, int]): The shape of the image.
        patch_size (int): The size of the patches.

    Returns:
        image (np.ndarray): The image of shape image_shape.
    """

    image = predictions.reshape(
        image_shape[0] - patch_size + 1, image_shape[1] - patch_size + 1
    )

    return image

def evaluate_segmentation(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluates the segmentation predictions.

    Args:
        preds (np.ndarray): The predicted shadow maps.
        labels (np.ndarray): The ground truth shadow maps.

    Returns:
        metrics (Dict[str, float]): A dictionary containing the metrics.
    """

    metrics = {
        "precision": 0.0,
        "dice": 0.0,
        "hausdorff": 0.0,
    }

    num_images = preds.shape[0]

    for i in range(num_images):

        pred = preds[i]
        label = labels[i]

        # Calculate precision
        tp = np.sum(np.logical_and(pred == 1, label == 1))
        fp = np.sum(np.logical_and(pred == 1, label == 0))
        precision = tp / (tp + fp + 1e-6)

        # Calculate dice coefficient
        dice = 2 * np.sum(np.logical_and(pred == 1, label == 1)) / (np.sum(pred) + np.sum(label) + 1e-6)

        # Calculate Hausdorff distance
        hd1 = directed_hausdorff(pred, label)[0]
        hd2 = directed_hausdorff(label, pred)[0]

        # The Hausdorff distance is the maximum of the two directed Hausdorff distances
        hd = float(max(hd1, hd2))

        metrics["precision"] += precision / num_images
        metrics["dice"] += dice / num_images
        metrics["hausdorff"] += hd / num_images

    return metrics
