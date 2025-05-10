#!/bin/bash

# remeber correct folder in stitch.launch

# Array of bag files to process
bag_files=("grad_nbv_pso" "grad_nbv_sub" "rand" "grid" "samp" "front" "grad_nbv")
# Array of frame numbers to use
frame_numbers=(10 20 25)

# Process each bag file with each frame number
for bag in "${bag_files[@]}"; do
  for frames in "${frame_numbers[@]}"; do
    echo "Processing bag: $bag with $frames frames"
    roslaunch mr-30 stitch.launch bag_file:=$bag number_of_frames:=$frames
    done
done
