import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# basic pan and zoom of any given image
# input_img
# centershift   with this set to 0--> zoom to the center of the image; otherwise shifted center of zoom
# pixel_step    Pixel steps between frames of the image list
# frameCount    Amount of frames

# Pan and zoom over input image by adjusting speed, centershift and the amount of frames
def pan_zoom(input_img, centershift_x=0, centershift_y=0, pixel_step=2, frames=25):
    img_list = []
    int_height = input_img.shape[0]
    int_width = input_img.shape[1]
    flt_ratio = float(int_width) / float(int_height)

    for i in range(0, frames):
        height1 = i * pixel_step + i * centershift_y
        height2 = int_height - i * pixel_step + i * centershift_y
        width1 = int(i * flt_ratio * pixel_step) + int(i * flt_ratio * centershift_x)
        width2 = int_width - int(i * flt_ratio * pixel_step) + int(i * flt_ratio * centershift_x)
        # limit height and width to image dimensions
        if height1 < 0:
            height1 = 0
        if height2 > int_height:
            height2 = int_height
        if width1 < 0:
            width1 = 0
        if width2 > int_width:
            width2 = int_width
        img = input_img[height1: height2, width1: width2]
        img = cv2.resize(img, (int_width, int_height), interpolation=cv2.INTER_AREA)
        img_list.append(img)
    return img_list


# get image and masks to generate frames
def add_parallax_masks(imgInput, imgBack, mask_list, centershift_x= 0, centershift_y= 0, step_obj=3, step_back=2, frames=25):
    obj_list = []  # list for the zoom video frame lists of the objects
    mask_zoom_list = []  # list for the zoom video frame lists of the masks
    vis_list = []  # list to visualize mask accuracy, not relevant to the script
    # Base for the animation is the zooming of the background with the slowest speed
    back_list= pan_zoom(input_img=imgBack, centershift_x=centershift_x, centershift_y=centershift_y,
                        pixel_step=step_back, frames=frames)
    # mask is [0] for background and [255] for object
    for i, mask_c in enumerate(mask_list):
        # When only 1 or 2 objects are detected adjust zoom speed
        if len(mask_list) <= 2:
            i = i+2
        step_object = step_back + step_obj + i  # Increased zooming speed for closer objects
        obj_image = np.zeros(imgInput.shape, dtype=np.uint8)
        obj_image[np.where(mask_c == 255)] = imgInput[np.where(mask_c == 255)]  # object pixels on black background
        vis_list.append(obj_image)  # Visualization
        # zoom the object with increasing speed for closer objects
        obj_zoom = pan_zoom(input_img=obj_image, centershift_x=centershift_x, pixel_step=step_object, frames=frames)
        # mask is also zoomed to use it for insertion on the background image
        mask_zoom = pan_zoom(input_img=mask_c, centershift_x=centershift_x, pixel_step=step_object, frames=frames)
        mask_zoom_list.append(mask_zoom)
        obj_list.append(obj_zoom)
    print("length of obj_list", len(obj_list))
    out_list = back_list
    # insert object pixels on the background starting from the furthest object to avoid overlaps with closer objects
    for obj, mask_z in zip(obj_list, mask_zoom_list):
        for i,frame in enumerate(out_list):
            new_frame = frame
            npy_Obj = obj[i]
            npy_Mask = mask_z[i]
            new_frame[npy_Mask == 255] = npy_Obj[npy_Mask == 255]
            out_list[i] = new_frame

    return out_list, vis_list
