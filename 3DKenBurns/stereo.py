import numpy as np
import cv2


# Function that Downsamples image reduce_factor number of times
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


# Function to get disparity map
def get_disparitymap(imgL, imgR):
    # Downsample each image 3 times (because they're too big)
    img_1_downsampled = downsample_image(imgL, 3)
    img_2_downsampled = downsample_image(imgR, 3)

    # Set disparity parameters
    win_size = 5 # winsize default 3; 5; odd number usually between 3...11
    min_disp = -1
    max_disp = 63  # min_disp * 9
    num_disp = max_disp - min_disp  # max_disp has to be dividable by 16 f. E. HH 192, 256

    # Create Block matching object
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=win_size,
                                   uniquenessRatio=5,
                                   speckleWindowSize=5,
                                   speckleRange=5,
                                   disp12MaxDiff=2,
                                   P1=8 * 3 * win_size ** 2,
                                   P2=32 * 3 * win_size ** 2)

    # Compute disparity map
    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

    # Normalize disparity map to 8 bit range
    normalizedDisp = cv2.normalize(src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    normalizedDisp = np.uint8(normalizedDisp)
    return normalizedDisp

