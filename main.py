import cv2
import os
from pathlib import Path
import animate
import novelView
import imageseg
import inpaint
import stereo
import numpy as np
from datetime import datetime



###############################################
# User input section
vertical_flag = True # True if top / bottom images on a vertical shift, False for horizontal shift
set_folder = 'set22'  # name of subfolder under path.home/Pictures/* with the stereo image set
img1_name = 'bottom0.jpg'  # left or bottom picture. Primary view used for image segmentation
img2_name = 'plus10.jpg'  # right or top picture

# Animation parameters
centershift_x = 0  # centershift in x direction; 0 means zoom to the image center
centershift_y = 0  # centershift in y direction
step_background = 3  # pixel step between animation frames, for faster zooming
step_object = 3  # additional object zooming speed to background; absolute object speed = step back + step object
frameCount = 25  # Frames created in novel view synthesis; 25 frames for 25 fps video results in a 1 second zoom in
                # and 1 second zoom back video = 2 sec video length


###############################################
# Path and filename configuration

filename = str(datetime.now().strftime("%Y%m%d-%H%M%S"))
# Stereo images input
STEREO_FOLDER = Path.home() / 'Pictures' / 'Stereo'
img1_path = str(STEREO_FOLDER / set_folder / img1_name)
img2_path = str(STEREO_FOLDER / set_folder / img2_name)

# Disparity map saved here
DISPARITY_FOLDER = Path.home() / 'Pictures' / 'disparitymaps'
disparity_path = str(DISPARITY_FOLDER / filename) +'_'+ set_folder +'.png'

# Segmented masks saved here
MASK_FOLDER = Path.home() / 'Pictures' / 'Masks'
local_mask_path = str(MASK_FOLDER / 'local' / filename) + '_' + set_folder
global_mask_path = str(MASK_FOLDER / 'global' / filename) + '_' + set_folder
dilated_mask_path = str(MASK_FOLDER / 'dilated' / filename) + '_' + set_folder

# Visualize segmented objects
VISUALIZE_FOLDER = Path.home() / 'Pictures' / 'Visualization'
vis_path = str(VISUALIZE_FOLDER / filename) + '_' + set_folder + '.png'

# Inpainted background saved here
INPAINT_FOLDER = Path.home() / 'Pictures' / 'Inpaint'
inpaint_path = str(INPAINT_FOLDER / filename) + '_' + set_folder +'.png'

# Animations saved here
ANIMATION_FOLDER = Path.home() / 'Videos' / 'Animations'
clip_name = str(ANIMATION_FOLDER / filename) + '_' + set_folder + '.mp4'
FRAME_FOLDER = ANIMATION_FOLDER / set_folder
Path(str(FRAME_FOLDER)).mkdir(parents=True, exist_ok=True)
frame_path = str(FRAME_FOLDER / filename)

###############################################
# Read images
npyImgL = cv2.imread(img1_path)
npyImgR = cv2.imread(img2_path)

print("img1 shape", npyImgL.shape)
print("img2 shape", npyImgR.shape)

###############################################
# Load camera parameters, get optimal camera matrix and undistort images
ret = np.load('/home/tobias/Documents/camera_params/ret.npy')
K = np.load('/home/tobias/Documents/camera_params/K.npy')
dist = np.load('/home/tobias/Documents/camera_params/dist.npy')
h,w = npyImgL.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w, h))
imgL_undistorted = cv2.undistort(npyImgL, K, dist, None, new_camera_matrix)
imgR_undistorted = cv2.undistort(npyImgR, K, dist, None, new_camera_matrix)
print("undistorted imgL shape",imgL_undistorted.shape)

###############################################
if vertical_flag == True:
    imgLdisp= cv2.rotate(imgL_undistorted, cv2.ROTATE_90_CLOCKWISE)
    imgRdisp = cv2.rotate(imgR_undistorted, cv2.ROTATE_90_CLOCKWISE)
else:
    imgLdisp = imgL_undistorted
    imgRdisp = imgR_undistorted
# Compute disparity map
disparity_map= stereo.get_disparitymap(imgL=imgLdisp, imgR=imgRdisp)
# rotate disparity map, because input images where rotated (top-down stereo pair rotate to left-right pair)
if vertical_flag == True:
    disparity_map = cv2.rotate(disparity_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite(disparity_path, disparity_map)


###############################################
# Image Segmentation to get object masks and write in MASK_FOLDER
mask_list, box_list, visualize_detect = imageseg.segmentation(img1_path)
for i,mask in enumerate(mask_list):
    cv2.imwrite(global_mask_path + '_mask%i.png' % i, mask)
visualize_detect = cv2.cvtColor(visualize_detect,cv2.COLOR_RGB2BGR)
cv2.imwrite(vis_path, visualize_detect)
print("length of mask list",len(mask_list))
print("length of box list",len(box_list))
print("shape of mask0",mask_list[0].shape)
print("coordinates of box0",box_list[0])

###############################################

# Correct Mask and fit to correct object position in the image
mask_list_c, mask_list_global = imageseg.mask_correction(imgInput=npyImgL, mask_list=mask_list, box_list=box_list,
                                                         iterations=10)
# sort masks by disparity
mask_sorted = imageseg.sort_objects(mask_list_global=mask_list_c, disparitymap= disparity_map)

###############################################
# Image inpainting
inpainted_image = inpaint.inpainting(imgInput=npyImgL, mask_list=mask_list_c)
#cv2.imwrite(inpaint_path, inpainted_image)

###############################################
# Add Parallax
multiobj_list, vis_mask_list = novelView.add_parallax_masks(imgInput=npyImgL, imgBack=inpainted_image,
                                             mask_list=mask_list_c, centershift_x=centershift_x,
                                             step_obj=step_object, step_back=step_background, frames=frameCount)
for i, vismask in enumerate(vis_mask_list):
    cv2.cvtColor(vismask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(VISUALIZE_FOLDER) + '/maskvis' + filename + set_folder + '_obj%i.png' % i, vismask)

###############################################
# Animate Clip from picture list
final_list = multiobj_list
# for i, frame in enumerate(final_list):
#     cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(frame_path + '_frame%i.png' % i, frame)

animate.gen_clip(images_list=(final_list+final_list[::-1]), clip_name=clip_name, fps=25)
