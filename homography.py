import cv2
import numpy as np
import sys

# Computes homography matrix between 2 sets of image points
def computeH(im1_pts, im2_pts):
    A = []

    # Creates A for Ah = 0 
    for (x_prime, y_prime), (x, y) in zip(im1_pts, im2_pts):
        A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)

    # This is a 9-element vector that is the flattened 
    # homography matrix
    h = Vt[-1, :]
    H = h.reshape(3, 3)

    # Normalize the matrix
    return H / H[-1, -1]


# Gets final homography matrix and inliers through ransac algorithm
def ransac(im1_pts, im2_pts, thresh=2.5, iter=3000):
    N = im1_pts.shape[0]
    most_inliers = []
    best_H = None

    for _ in range(iter):
        # Randomly pick 4 point correspondences
        index = np.random.choice(N, 4, replace=False)
        ref_points_sample = im1_pts[index]
        points_to_move_sample = im2_pts[index]

        # Compute H
        H = computeH(ref_points_sample, points_to_move_sample)

        # Project im2 onto the reference
        proj = project(H, im2_pts)

        # Compute error and get list of inliers
        errors = np.linalg.norm(proj - im1_pts, axis=1)
        inliers = np.where(errors < thresh)
        inliers = inliers[0]

        # Saves inlier list if it's the largest number of inliers
        if len(inliers) > len(most_inliers):
            most_inliers = inliers
            best_H = H
        
    # Recompute homography with all inliers
    if len(most_inliers) >= 4:
        H_final = computeH(im1_pts[most_inliers], im2_pts[most_inliers])
    
    else:
        H_final = best_H
    
    return H_final, most_inliers


# Projects all points in an image by H matrix
def project(H, pts):
    # Converts points to homogeneous coordinates (adds a 1)
    pts_homog = np.hstack([pts, np.ones((pts.shape[0], 1))])

    # proj = H * [x y 1] for each point
    proj = (H @ pts_homog.T).T

    # Divide x and y by w
    proj = proj[:, :2] / proj[:, 2, None]
    
    return proj


 # Gets keypoints and descriptors for each image
def get_features(im1, im2):
    sift = cv2.SIFT_create()

    im1_keypoints, im1_descriptors = sift.detectAndCompute(im1, None)
    im2_keypoints, im2_descriptors = sift.detectAndCompute(im2, None)

    return im1_keypoints, im1_descriptors, im2_keypoints, im2_descriptors


def get_feature_matches(im1, im2):
    # Gets keypoints and descriptors in images
    im1_keypoints, im1_descriptors, im2_keypoints, im2_descriptors = get_features(im1, im2)

    # Match keypoints
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(im1_descriptors, im2_descriptors, k=2)

    # thresholding
    threshold = 0.67
    good_matches = []
    for m, n in matches:
        ratio = m.distance / (n.distance + 1e-8)
        if ratio < threshold:
            good_matches.append(m)
    
    return good_matches, im1_keypoints, im2_keypoints



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <ref_image> <image_to_move> <H-path>")
        exit(1)

    # Reads in images
    ref_image = cv2.imread(sys.argv[1])
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    next_image = cv2.imread(sys.argv[2])
    next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)

    # Gets thresholded feature matches 
    matches, ref_keypoints, next_keypoints = get_feature_matches(ref_image, next_image)

    # Gets indices of matches
    ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_to_move = np.float32([next_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Gets homography matrix and inliers
    H, inliers = ransac(ref_pts, pts_to_move)

    np.save(sys.argv[3], H)