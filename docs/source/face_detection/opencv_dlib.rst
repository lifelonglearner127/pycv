OpenCV, Dlib Face Detection
===========================


OpenCV Viola-Jones
------------------

Running the script::

    python detect_image.py --image <image_path>
    python detect_video.py --image <video_path>
    python detect_camera.py


OpenCV DNN
----------

Running the script::

    python detect_image.py --image <image_path> --method 3
    python detect_video.py --image <video_path> --method 3
    python detect_camera.py --method 3


Dlib HOG
--------

Running the script::

    python detect_image.py --image <image_path> --method 1
    python detect_video.py --image <video_path> --method 1
    python detect_camera.py --method 1
    

Dlib CNN
--------

Running the script::

    python detect_image.py --image <image_path> --method 2
    python detect_video.py --image <video_path> --method 2
    python detect_camera.py --method 2
