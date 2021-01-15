This tutorial can be used to calibrate industrial cameras which are used to process images with OpenCV.

This way we can transform images from 
![Distortions](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/Calibrate_Camera_with_Chessboard/distortion.jpg)

to
![Undistorted](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/Calibrate_Camera_with_Chessboard/corrected_img.jpg)

In order to calibrate the intrinsic parameters of a camera and remove fish eye disturbances etc. it is neccessary to do the following steps 

1. print out the [chessboard](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/Calibrate_Camera_with_Chessboard/pattern.png) in any size and glue it to a flat surface.
2. place the chessboard before the camera as close as possible and take a photo in the resolution the camera will operate in your application. The generated matrix is only valid for the exact same resolution (e.g. 1920x1080)
3. adpat the path in the [calibration_.py](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/Calibrate_Camera_with_Chessboard/calibration_2.py) file to the before taken image and run the script in the command line with "python3 calibration_.py"
4. The script generates a calibration_data.pkl file, which has to be loaded everytime the camera is used. How the calibration_data can be applied to the camera can be seen in the script [test_calibration_on_images.py](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/Calibrate_Camera_with_Chessboard/test_calibration_on_images.py)
