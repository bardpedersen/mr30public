#!/bin/bash

# Define methods and frame numbers to process
METHODS=("grad_nbv_pso" "grad_nbv_sub" "rand" "grid" "samp" "front" "grad_nbv")
FRAMES=("05" "10" "15" "20" "25")

# Create output directory if it doesn't exist
mkdir -p mr30_ws/data/2504_realsense/metric/

# Loop through each combination and run evaluator
for method in "${METHODS[@]}"; do
    for frame in "${FRAMES[@]}"; do
        INPUT_PCD="mr30_ws/data/2504_realsense/pcd/${method}${frame}_cropped.pcd"
        OUTPUT_DIR="mr30_ws/data/2504_realsense/metric/${method}${frame}"
        REFERENCE_PCD="mr30_ws/data/2504_realsense/pcd/armadillo.pcd"

        # Check if input PCD exists
        if [ -f "$INPUT_PCD" ]; then
            echo "Processing: ${method}${frame}"
            echo "Command: rosrun mr-30 pointcloud_evaluator $REFERENCE_PCD $INPUT_PCD $OUTPUT_DIR"
            
            # Execute the command
            rosrun mr-30 pointcloud_evaluator $REFERENCE_PCD $INPUT_PCD $OUTPUT_DIR
            
            # Add a small delay to avoid overwhelming the system
            sleep 1
        else
            echo "Skipping ${method}${frame}: Input file not found: $INPUT_PCD"
        fi
    done
done

echo "All evaluations complete!"
