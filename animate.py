import moviepy.editor
import cv2


def gen_clip(images_list, clip_name, fps):
    out_list= []
    for image in images_list:
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out_list.append(image)
    clip = moviepy.editor.ImageSequenceClip(sequence=out_list, fps=fps)
    clip.write_videofile(clip_name, fps=fps)
    return 0
