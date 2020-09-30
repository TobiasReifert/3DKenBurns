# 3DKenBurns

This code can animate the 3D Ken Burns effect based on stereo image input.

The code is the result of a Bachelor's thesis project.

**Abstract of the thesis:**<br>
The creation of 3D effects out of 2D images is mostly inaccessible for amateurs without professional photo editing software or specialized hardware for depth measurement.
In this thesis, a method that achieves 3D measurement to automatically synthesize the 3D Ken Burns effect just using a single smartphone camera sensor is proposed. First, two images of the same scene from different viewing positions are captured. The camera movement is restrained to one axis by using a slide bar. Second, the stereo pictures are transferred to an image processing pipeline consisting of disparity calculation to estimate image depth, instance segmentation with a neural network, image inpainting for the background and novel view synthesis. Finally, combining the obtained scene information the 3D Ken Burns effect animation is rendered to a video file. As opposed to other solutions, this method does not require professional skills or setups and can run on common smartphones, while neural network predictions being outsourced to a cloud. Therefore it has the potential to become an accessible automated technology for everyday use.


----------------------
## Instructions
### 1. Calibration
**This step is only necessary if you use your own stereo pictures and not the examples.**

1. Shoot chessboard calibration images with your camera and save them to a folder on your PC.

2. Adjust the user input section at the top of ``calibration.py``. Parameter meanings are explained below.

3. Run ``calibration.py`` file to obtain camera parameters and remove distortion.

### 2. Take stereo images

Example images are provided in ``examples/``. The default calibration parameters are adjusted to the camera setup used with the example pictures.

You can also use your own pictures:<br>
Shoot stereo images with a baseline of approx. 5 to 15 cm for object distances between 1 and 5 meter.
The rule of thumb is choosing 1/30 to 1/20 of the closest object distance as the baseline. Since the images wont be rectified, it is important to avoid rotation and lateral shift between pictures. Best practice is to mount the camera to a slide bar which can be used vertical or horizontal, for best disparity map results.

### 3. Animate 3D Ken Burns

1. Adjust the paths to the stereo image files in the user input section on top of ``main.py``. For explanations of parameters, see below.

2. **[optional]**  Adjust animation parameters to alter the pan and zoom effect. For explanations of parameters, see below.

3. Execute ``main.py`` and the results are saved in the ``outputs/`` folder.



#### Parameters

For animation:
- ``vertical_flag``: Specifies direction of camera movement. ``True`` if the camera moved vertically, and ``False`` if it moved horizontally between taking the two pictures.

- ``centershift``: Number of pixels the image center moves between individual video frames during the effect (specified for x and y separately)

- ``step_background``: Zooming speed of background layer in pixels per video frame

- ``step_object``: Zooming speed of objects in pixels per video frame (in addition to background movement)

- ``frame_count``: Amount of video frames. Video runs at 25 frames per second.

For calibration:
- ``chessboard_size``: Size of the chessboard, specified in number of internal corners between the squares.

- ``path_to_calibration_images``: Path to all calibration images. It is recommended to put all in one folder and use a wildcard, e.g. ``calibration_imgs/*``


----------------------
## Setup

1. Install ``3DKenBurns.yml`` using ``conda env create -f 3DKenBurns.yml``.

2. Activate enviroment with ``conda activate 3DKenBurns``.

3. Install [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) into the environment. To do this, clone the repository and run ``pip install . --user``.

4. Since ``moviepy`` expects an older version of FFMPEG, a soft link for has to be created:<br>
``ln -s ~/anaconda3/envs/3DKenBurns/lib/libopenh264.so ~/anaconda3/envs/3DKenBurns/lib/libopenh264.so.5``

5. Load the ``resnet50_coco_v0.2.0.h5`` weights from the [fizyr/keras-maskrcnn realease page](https://github.com/fizyr/keras-maskrcnn/releases) and store them in ``3DKenBurns/resnet50_weights/``.


----------------------
## Author

Tobias Reifert, University of Applied Sciences Karlsruhe
