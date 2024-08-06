import argparse
import os

from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check the shapes of the data in the dataset.")
    parser.add_argument("data_dir", type=str, help="The directory containing the dataset.")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    
    # Load a single image to get the shape
    image = Image.open(os.path.join(images_dir, "0.png"))

    # Get the shape of the image
    width, height = image.size

    # Get the number of images
    num_images = len(os.listdir(images_dir))

    print(f"Widht: {width}, Height: {height}, Number of images: {num_images}")