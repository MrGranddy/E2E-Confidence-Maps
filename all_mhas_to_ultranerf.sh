#!/bin/bash

# Directory containing the MHA files.
input_dir="/home/vanessa_share/bugra/dataset_project/data/mhas"

# Main output directory for all converted data.
main_output_dir="/home/vanessa_share/bugra/dataset_project/data/ultranerf_data"

# Python script path.
python_script_path="/home/vanessa_share/bugra/dataset_project/mha_to_ultranerf.py"

# Loop through all MHA files in the input directory.
for file in "$input_dir"/*.mha; do
    # Extract the filename without the path and extension.
    filename=$(basename -- "$file" .mha)

    # Use awk to separate the filename into the needed parts for the output directory.
    # Assuming format is CAT_THx1.mha, we extract CAT_TH.
    output_subdir=$(echo "$filename" | awk -F'x' '{print $1}')

    # Combine main output directory with the extracted part of the filename.
    output_path="$main_output_dir/$output_subdir"

    # Execute the Python script with the input file and output directory.
    python "$python_script_path" "$file" "$output_path"

    echo "Processed $file and saved output to $output_path"
done

echo "All files have been processed."
