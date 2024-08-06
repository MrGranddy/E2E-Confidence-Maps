import argparse
import math
import os

import numpy as np

from PIL import Image

import SimpleITK as sitk
import nibabel as nib


def create_inverse_transform(transform):
    """Computes the inverse of a given transformation matrix.

    Args:
        transform (np.ndarray): The original 4x4 transformation matrix.

    Returns:
        np.ndarray: Inverted 4x4 transformation matrix.
    """
    return np.linalg.inv(transform)


def apply_inverse_transform(poses, inverse_transform):
    """Applies an inverse transformation matrix to a set of poses.

    Args:
        poses (np.ndarray): Original poses.
        inverse_transform (np.ndarray): Inverse transformation matrix.

    Returns:
        np.ndarray: Transformed poses.
    """
    return (inverse_transform @ poses.transpose(0, 2, 1)).transpose(0, 2, 1)


def load_poses_and_images(input_dir):
    """Loads poses and images from the specified directory.

    Args:
        input_dir (str): The directory containing the images and poses files.

    Returns:
        tuple: Tuple containing image data (3D numpy array) and poses (4D numpy array).
    """
    images = np.load(os.path.join(input_dir, "images.npy"))
    poses = np.load(os.path.join(input_dir, "poses.npy"))
    with open(os.path.join(input_dir, "spacing.txt"), "r") as f:
        spacing = tuple(map(float, f.read().strip().split()))

    return images, poses, spacing


def save_as_mha(file_path, image_data, poses, spacing):
    """Saves image data and poses as an MHA file.

    Args:
        file_path (str): Path to save the MHA file.
        image_data (np.ndarray): The image data as a 3D numpy array.
        poses (np.ndarray): The poses as a 4D numpy array.
        spacing (tuple): The spacing of the image data.
    """
    image = sitk.GetImageFromArray(image_data)
    image.SetSpacing(spacing)

    for i, pose in enumerate(poses):
        transform_key = f"Seq_Frame{i:04d}_ImageToReferenceTransform"
        transform_line = ' '.join(map(str, pose.flatten().tolist()))
        image.SetMetaData(transform_key, transform_line)

    sitk.WriteImage(image, file_path)


def main(args):
    # Load images and poses from the numpy files.
    image_data, poses, spacing = load_poses_and_images(args.input)

    # Compute the inverse transformation
    inverse_transform = create_inverse_transform(create_linear_transform(0, 0, 0, 0, -30, -180))

    # Apply the inverse transformation to the poses
    poses = apply_inverse_transform(poses, inverse_transform)

    # Save as MHA file
    save_as_mha(args.output, image_data, poses, spacing)

    print(f"Converted UltraNerf dataset to MHA and saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts an UltraNerf dataset to an MHA file.")
    parser.add_argument("input", type=str, help="The input directory containing the UltraNerf dataset.")
    parser.add_argument("output", type=str, help="The output MHA file.")

    args = parser.parse_args()

    main(args)
