#!/usr/bin/env bash

# Makes sure correct number of arguments given
if [ "$#" -ne 4 ]; then
    echo "Wrong number of arguments"
    echo "Usage: $0 <left_image> <middle_image> <right_image> <mosaic_file_name>"
    exit 1
fi

left_image="$1"
ref_image="$2"
right_image="$3"
mosaic_name="$4"

python homography.py $ref_image $left_image H12
python homography.py $ref_image $right_image H23
python stitching_3.py $left_image $ref_image $right_image H12.npy H23.npy $mosaic_name

rm H12.npy H23.npy