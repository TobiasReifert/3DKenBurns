import glob

import cv2
import numpy as np
from tqdm import tqdm

###############################################
# User input section

# Modify chessboard size. The size is measured in internal corners.
chessboard_size = (7, 9)

# Read images, set path to Calibration set folder
path_to_calibration_images = '../../Pictures/Calibration/set2/*'
calibration_paths = glob.glob(path_to_calibration_images)

###############################################

# Define arrays to save detected points
obj_points = []  # 3D points in real world space
img_points = []  # 3D points in image plane
# Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# define criteria for subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Iterate over images to find intrinsic matrix
for image_path in tqdm(calibration_paths):
    # Load image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    if ret == True:
        print("Chessboard detected!")
        print(image_path)
        # refine corner location (to subpixel accuracy) based on criteria.
        corners = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        image = cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None, None)

# Save parameters into numpy file
np.save("config/K", K)
np.save("config/dist", dist)
