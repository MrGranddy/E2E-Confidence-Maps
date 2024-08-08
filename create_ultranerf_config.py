import os
import sys

from PIL import Image

# Extract dataset directory and number of samples from command line arguments
dataset_path = sys.argv[1]
logdir = sys.argv[2]

# Extract dataset name from the last part of the path
dataset_name = os.path.basename(os.path.normpath(dataset_path))

images_dir = os.path.join(dataset_path, "images")

# Load a single image to get the shape
image = Image.open(os.path.join(images_dir, "0.png"))

# Get the shape of the image
width, height = image.size

# N_samples is the height of the image
N_samples = height

# Define the content of the config file
config_content = f"""
expname = {dataset_name}
loss = ssim
basedir = {logdir}
datadir = {dataset_path}
dataset_type = us
no_batching = True
output_ch = 5

N_samples = {N_samples}

i_embed = 6
probe_depth = 93
probe_width = 38
"""

# Path for the temporary config file
config_filename = f"{dataset_name}_config.txt"

# Write the config content to the file
with open(config_filename, "w") as config_file:
    config_file.write(config_content)

# Print the absolute path of the config file
print(os.path.abspath(config_filename))
