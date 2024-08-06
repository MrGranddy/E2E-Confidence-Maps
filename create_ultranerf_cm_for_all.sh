#!/bin/bash

# This parameter decides which subset of directories to process: 0, 1, or 2
subset_index=$1

# Directory containing datasets
logdir="/home/vanessa_share/bugra/dataset_project/logs"
confidence_map_dir="/home/vanessa_share/bugra/dataset_project/confidence_maps"

# Gather all dataset directories into an array
log_dirs=($logdir/*)

# Calculate total number of directories
total_dirs=${#log_dirs[@]}

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
    log_dir=${log_dirs[$i]}
    # Extract only the directory name
    dataset_name=$(basename "$log_dir")

    # Define the path to the specific subdirectory for processing
    params_path="${log_dir}/output_maps_${dataset_name}_model_099000_0/params"

    echo "Processing dataset: $dataset_name at $params_path"

    python render_to_confidence_map.py --input_path "$params_path" --participant_name "$dataset_name" --destination_dir "$confidence_map_dir"

done
