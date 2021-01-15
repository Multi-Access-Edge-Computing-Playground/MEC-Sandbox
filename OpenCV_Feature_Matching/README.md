Feature Matching is a more complex algorithm (but no machine learning) to detect objects and is not based primarily on the contour but rather on special features like edges. This way the orientation of an object can be 
detected more exact which comes in handy when the robot is supposed to grab a specific part/region of the object.. The SIFT-algorithm that is used there is patented, so carefull with using it in commercial applications. 
Use the script [feature_matching_draw_matches.py](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/OpenCV_Feature_Matching/feature_matching_draw_matches.py)
to test a feature matching algorithm with a custom object.
Do the following steps:
1. Take a picture of the object and crop it so that only the object is visible. This is the "query" image. Then take an image of the cluttered image where the object is located (maybe with other objects and hard to detect)
- this is the "train" image.
2. Specify the path to both images in the script and execute it. The script displays only the matched features.

To extract the bounding Box of the object in the "train" image, use the script [feature_matching_w_bounding_box.py](https://github.com/Multi-Access-Edge-Computing-Playground/MEC-Sandbox/blob/master/OpenCV_Feature_Matching/feature_matching_w_bounding_box.py)
The pixel coordinates can then be processed into robot coordinates (comming soon)

To detect multiple objects, use the 
