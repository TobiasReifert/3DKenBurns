# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# imports
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


# Dilate mask to make sure the object is contained in the mask for inpainting
# Also fit the mask to image size and to the global position
def mask_correction(imgInput, mask_list, box_list, iterations=10):
    mask_list_global_dil= []  # Mask list with dilated masks for better inpainting
    mask_list_global= []  # Mask list positioned in the image frame
    kernel = np.ones((5, 5), np.uint8)
    kernel.fill(255)
    for (mask, box) in zip(mask_list, box_list):
        print(box)
        print(mask)
        mask_image = np.zeros((imgInput.shape[0], imgInput.shape[1]), np.uint8)
        mask_image[box[1]:box[3], box[0]:box[2]] = mask
        mask_list_global.append(mask_image)
        mask_dilated = cv2.dilate(mask_image, kernel, iterations=iterations)
        mask_list_global_dil.append(mask_dilated)
    return mask_list_global_dil, mask_list_global


def sort_objects(mask_list_global, disparitymap):
    mean_list = []
    print(disparitymap.shape)
    for mask in mask_list_global:
        mask = cv2.resize(mask,disparitymap.shape[::-1])
        #disp= np.zeros(disparitymap.shape, np.uint8)
        disp = np.zeros(disparitymap.shape, np.float)
        disp[mask == 255] = disparitymap[mask == 255]
        disp[mask != 255] = np.nan
        mean = np.nanmean(disp)
        mean_list.append(mean)
        print("unsorted", mask[0])
        print(mean_list)
    mask_list = np.array(mask_list_global)
    mean_list = np.array(mean_list)
    inds = mean_list.argsort()
    print("Mean values sorted:", inds)
    mask_sorted = mask_list[inds]
    for mask in mask_sorted:
        print("sorted", mask[0])
    return mask_sorted


# Function for instance segmentation for a given filepath based on the example jupyter notebook of the
# Fizyr/keras-maskrcnn github repo
def segmentation(imgpath, score_threshold = 0.5, binarize_threshold= 0.5):
    # Setting gpu to 0
    setup_gpu(0)
    # adjust this to point to your downloaded/trained model
    # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_v0.2.0.h5')
    model_path = os.path.join('/home/tobias/Bachelorarbeit/keras-retinanet/', 'snapshots', 'resnet50_coco_v0.2.0.h5')
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    # print(model.summary())
    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                       7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                       12: 'parking meter',
                       13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                       20: 'elephant',
                       21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                       28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                       34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                       46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                       52: 'hot dog',
                       53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                       60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                       66: 'keyboard',
                       67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                       73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                       79: 'toothbrush'}
    # load image
    image = read_image_bgr(imgpath)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    # process image
    start = time.time()
    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    boxes = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks = outputs[-1][0]
    print("box shape", boxes.shape)
    print("scores shape", scores.shape)
    print("labels shape", labels.shape)
    print("masks shape",masks.shape)
    # correct for image scale
    boxes /= scale
    mask_list = []
    box_list = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < 0.5:
                break
        # save box coordinates in list
        box = box.astype(np.int16)
        box_list.append(box[:])
        # resize to fit the box
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

        # binarize the mask
        mask = (mask > binarize_threshold).astype(np.uint8)
        mask = cv2.normalize(src=mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_list.append(mask[:,:,label])

    # visualize detections
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        mask = mask[:, :, label]
        draw_mask(draw, b, mask, color=label_color(label))

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    return mask_list, box_list, draw

