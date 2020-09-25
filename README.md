# 3DKenBurns
The creation of 3D effects out of 2D images is mostly inaccessible for amateurs without professional photo editing software or specialized hardware for depth measurement.
In this thesis, a method that achieves 3D measurement to automatically synthesize the 3D Ken Burns effect just using a single smartphone camera sensor is proposed. First, two images of the same scene from different viewing positions are captured. The camera movement is restrained to one axis by using a slide bar. Second, the stereo pictures are transferred to an image processing pipeline consisting of disparity calculation to estimate image depth, instance segmentation with a neural network, image inpainting for the background and novel view synthesis. Finally, combining the obtained scene information the 3D Ken Burns effect animation is rendered to a video file. As opposed to other solutions, this method does not require professional skills or setups and can run on common smartphones, while neural network predictions being outsourced to a cloud. Therefore it has the potential to become an accessible automated technology for everyday use.

## How to
First, shoot chessboard calibration images with your camera and safe them to a folder on your PC. See calibration.py and modify chessboard_size and calibration_paths parameter accordingly. Run calibration.py file to obtain camera parameters for undistortion.

Second, shoot stereo images with a baseline of approx. 5 to 15 cm baseline for object distances between 1 and 5 meter. Since the images wont be rectified, it is important to avoid rotation and lateral shift between pictures. Best practice is using a slide bar which can be used vertical or horizontal, for best disparity map results. For horizontal shift set vertical flag = False, otherwise true for vertical shift. Under the user input section the path to the stereo pair is modified. To configure the virtual camera scan, set the animation parameter section. Set the path and filename configuration section according to your folder structure. Standard path for stereo sets: "/home/user/Pictures/Stereo/". Also modify the path if necessary to the saved camera parameters obtained prior.
Finally, run the main.py and the 3D Ken Burns video will be saved in MPEG-4 format to the standard path "/home/user/Videos/Animation/".

## Requirements
Mostly implemented with Open CV functions, see requirements.txt for required packages.
Instance segmentation based on Mask R-CNN implementation by https://github.com/fizyr/keras-maskrcnn, see for further requirements

The path to ffmpeg might need to be set, e.g.: " os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/user/anaconda3/envs/ENVIROMENTNAME/bin/ffmpeg"
