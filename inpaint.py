import cv2


def inpainting(imgInput, mask_list):
    dst= imgInput
    for mask_c in mask_list:
        dst = cv2.inpaint(dst, mask_c, 3, cv2.INPAINT_TELEA)
    return dst
