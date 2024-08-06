#!/bin/bash

# This parameter decides which subset of directories to process: 0 to 5
subset_index=$1

# Directory containing datasets
datasetdir="/home/vanessa_share/bugra/dataset_project/data/ultranerf_data"
confidence_map_dir="/home/vanessa_share/bugra/dataset_project/confidence_maps"

# Gather all dataset directories into an array
dataset_dirs=($datasetdir/*)

# Calculate total number of directories
total_dirs=${#dataset_dirs[@]}

# Calculate the number of directories each script should process, rounding up if not divisible exactly
dirs_per_script=$(( (total_dirs + 5) / 6 )) # Ensures all directories are covered by rounding up

# Calculate the starting and ending index for the subset
start_index=$((dirs_per_script * subset_index))
end_index=$((start_index + dirs_per_script - 1))

# Adjust the end index for the last subset to ensure it doesn't exceed the number of directories
if [ $subset_index -eq 5 ]; then
    end_index=$((total_dirs - 1))
fi

# Iterate over each dataset directory in the subset
for (( i=start_index; i<=end_index; i++ )); do
    dataset_dir=${dataset_dirs[$i]}
    # Extract only the directory name
    dataset_name=$(basename "$dataset_dir")

    # Define the path to the specific subdirectory for processing
    images_path="${dataset_dir}/images"

    # Check if the .npy file already exists in the destination directory
    if [ -f "${confidence_map_dir}/acyclic/${dataset_name}.npy" ]; then
        echo "${dataset_name}.npy already exists. Skipping..."
        continue # Skip the rest of the loop if file exists
    fi

    echo "Processing dataset: $dataset_name at $images_path"

    python create_acyclic_cm.py --input_path "$images_path" --participant_name "$dataset_name" --destination_dir "$confidence_map_dir"

done
