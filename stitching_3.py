import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(f"Usage: python {sys.argv[0]} <image_1> <ref-image> <image_3> <H12-path> <H23-path> <mosaic-name>")
        exit(1)
    
    # Get images
    left_image = cv2.imread(sys.argv[1])
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

    ref_image = cv2.imread(sys.argv[2])
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    right_image = cv2.imread(sys.argv[3])
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    # Load homography matrices
    H_12 = np.load(sys.argv[4])
    H_23 = np.load(sys.argv[5])

    mosaic_name = sys.argv[6]

    # Dimensions of mosaic before cropping
    h, w = left_image.shape[:2]
    mosaic_width = w*3
    mosaic_height = h*2

    # Initializes empty mosaic
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    # Place reference image in center of mosaic
    offset_x = w     
    offset_y = h // 2

    mosaic[offset_y:offset_y+h, offset_x:offset_x+w] = ref_image

    # Matrix to translate by the offset
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # Translated homographies
    H_left = T @ H_12
    H_right = T @ H_23

    # Place left image
    warped_left = cv2.warpPerspective(left_image, H_left, (mosaic_width, mosaic_height))
    mask_L = (warped_left > 0)
    mosaic[mask_L] = warped_left[mask_L]

    # Place right image
    warped_right = cv2.warpPerspective(right_image, H_right, (mosaic_width, mosaic_height))
    mask_R = (warped_right > 0)
    mosaic[mask_R] = warped_right[mask_R]

    # Get artifact-free rectangle and crop
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    foreground = gray > 0

    coords = np.column_stack(np.where(foreground))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = mosaic[offset_y:offset_y + h, x_min:x_max]
    plt.imsave(mosaic_name, cropped)
