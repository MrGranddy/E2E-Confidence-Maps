#!/bin/bash

# This parameter decides which subset of directories to process: 0, 1, or 2
subset_index=$1

# Directory containing datasets
datasets_dir="/home/vanessa_share/bugra/dataset_project/data/ultranerf_data"
logdir="/home/vanessa_share/bugra/dataset_project/logs"

# Gather all dataset directories into an array
dataset_dirs=($datasets_dir/*)

# Calculate total number of directories
total_dirs=${#dataset_dirs[@]}

# Calculate the number of directories each script should process
dirs_per_script=$((total_dirs / 3))

# Calculate the starting and ending index for the subset
start_index=$((dirs_per_script * subset_index))
end_index=$((start_index + dirs_per_script - 1))

# If it's the last subset, include any remaining directories
if [ $subset_index -eq 2 ]; then
    end_index=$((total_dirs - 1))
fi

# Iterate over each dataset directory in the subset
for (( i=start_index; i<=end_index; i++ )); do
    dataset_dir=${dataset_dirs[$i]}
    # Extract only the directory name
    dataset_name=$(basename "$dataset_dir")
    echo "Processing dataset: $dataset_name"

    cd /home/vanessa_share/bugra/ultra-nerf-private
    python render_demo_us.py --logdir "$logdir" --expname "$dataset_name" --model_epoch 99000
    cd /home/vanessa_share/bugra/dataset_project

done
