# 3DKenBurns

This code animates the 3D Ken Burns effect for still images based on stereo image input.

It is the result of a Bachelor's thesis project with the title <br>
_"Smartphone Photography: Creating the 3D Ken Burns Effect by using Monocular Stereo Vision, Instance Segmentation and Image Inpainting"_.

**Abstract of the thesis:**<br>
The creation of 3D effects out of 2D images is mostly inaccessible for amateurs without professional photo editing software or specialised hardware for depth measurement.
In this thesis, a method that achieves 3D measurement to automatically synthesize the 3D Ken Burns effect just using a single smartphone camera sensor is proposed. First, two images of the same scene from different viewing positions are captured. The camera movement is restrained to one axis by using a slide bar. Second, the stereo pictures are transferred to an image processing pipeline consisting of disparity calculation to estimate image depth, instance segmentation with a neural network, image inpainting for the background and novel view synthesis. Finally, combining the obtained scene information the 3D Ken Burns effect animation is rendered to a video file. As opposed to other solutions, this method does not require professional skills or setups and can run on common smartphones, while neural network predictions being outsourced to a cloud. Therefore it has the potential to become an accessible automated technology for everyday use.


## Setup
On a Unix-based system, follow these steps:

1. Clone this repository and `cd` into the folder.

2. Create a new conda environment from ``3DKenBurns.yml`` using ``conda env create -f 3DKenBurns.yml``.

3. Switch to the new enviroment with ``conda activate 3DKenBurns``.

4. Since ``moviepy`` expects an older version of FFMPEG with different file names, a soft link has to be created:<br>
``ln -s ~/anaconda3/envs/3DKenBurns/lib/libopenh264.so ~/anaconda3/envs/3DKenBurns/lib/libopenh264.so.5`` <br>
The paths might have to be adjusted, depending on the location of your conda environment on your system.

5. Download the ``resnet50_coco_v0.2.0.h5`` weights from the [fizyr/keras-maskrcnn realease page](https://github.com/fizyr/keras-maskrcnn/releases):<br>
First create their folder with `mkdir 3DKenBurns/resnet50_weights`. Then put the weights file into this folder.

6. In line 4 of `3DKenBurns/main.py`, adjust the path to FFMPEG depending on the location of your conda environment on your system.


## Instructions
### 1. Calibration
_This step is only necessary if you use your own stereo pictures and not the provided examples._

1. Shoot chessboard calibration images with your camera and save them to a folder on your computer.

2. Adjust the user input section at the top of ``3DKenBurns/calibration.py``. Parameters are explained below.

3. Go into the ``3DKenBurns/`` folder and execute ``calibration.py`` file to obtain camera parameters necessary for removing distortion.

### 2. Take stereo images

Example images are provided in ``examples/``. The default calibration parameters are adjusted to the camera setup used with the example pictures.

You can also use your own pictures:<br>
Shoot stereo images with a baseline of approximately 5 to 15 cm for object distances between 1 and 5 meters.
The rule of thumb is choosing 1/30 to 1/20 of the closest object distance as the baseline. Since the images won't be rectified, it is important to avoid rotation and lateral shift between pictures. The best practice is to mount the camera to a slide bar which can be used vertically or horizontally, for optimised disparity map results.

### 3. Animate 3D Ken Burns
_If you want to use the provided example pictures, you can skip steps 1 and 2._

1. Adjust the paths to the stereo image files in the user input section at the top of ``3DKenBurns/main.py``.

2. **[optional]**  Adjust animation parameters to alter the pan and zoom effect. For explanations of parameters, see below.

3. Go into the `3DKenBurns/` folder and execute ``main.py``. The results will be saved in the ``outputs/`` folder.

--------------------

### Parameters

For calibration:
- ``chessboard_size``: Size of the chessboard, specified in number of internal corners between the squares.

- ``path_to_calibration_images``: Path to all calibration images. It is recommended to put all in one folder and use a wildcard, e.g. ``calibration_imgs/*``

For animation:
- ``vertical_flag``: Specifies direction of camera movement. ``True`` if the camera moved vertically, and ``False`` if it moved horizontally between taking the two pictures.

- ``centershift``: Number of pixels the image center moves between individual video frames during the effect (specified for x and y separately)

- ``step_background``: Zooming speed of background layer in pixels per video frame

- ``step_object``: Zooming speed of objects in pixels per video frame (in addition to background movement)

- ``frame_count``: Amount of video frames. Video runs at 25 frames per second.



## Author

Tobias Reifert, University of Applied Sciences Karlsruhe
