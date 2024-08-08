#!/bin/bash

# This parameter decides which subset of directories to process: 0, 1, or 2
subset_index=$1

# Directory containing datasets
datasets_dir="/home/vanessa_share/bugra/dataset_project/data/ultranerf_data"
logdir="/home/vanessa_share/bugra/dataset_project/logs"

# Python script path
python_script_path="/home/vanessa_share/bugra/dataset_project/create_ultranerf_config.py"

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
    # Create the temporary config file and capture the filename
    config_file=$(python "$python_script_path" "$dataset_dir" "$logdir")

    # cat config_file
    cat "$config_file"

    cd /home/vanessa_share/bugra/ultra-nerf-private
    python run_ultra_nerf.py --config "$config_file" --n_iters 100000
    cd /home/vanessa_share/bugra/dataset_project

    # Remove the temporary config file after training
    rm "$config_file"
done
