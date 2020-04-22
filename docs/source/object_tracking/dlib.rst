Dlib Object Tracking
====================


People Counter
--------------

- Utilizes deep learning object detectors for improved person detection accuracy
- Leverages two separate object tracking algorithms, including both centroid tracking and correlation filters for improved tracking accuracy
- Applies both a “detection” and “tracking” phase, making it capable of (1) detecting new people and (2) picking up people that may have been “lost” during the tracking phase

Running the script::

    python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/people_counting.mp4


Single Object Tracking
----------------------
dlib’s correlation tracking algorithm is quite robust and capable of running in real-time.
However, the biggest drawback is that the correlation tracker can become “confused” and lose the object we wish to track if viewpoint changes substantially or if the object to be tracked becomes occluded.

Running the script::

    python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person


Multiple Object Tracking
------------------------
This example is a simple extension to the above example. It tracks multiple objects at the same time. And it runs on only one process.

Running the script::

    python multiple_track_object_nmp.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person
    

Fast Multiple Object Tracking
-----------------------------
In order to speed up our object tracking pipeline we can leverage Python’s multiprocessing module, similar to the threading module, but instead used to spawn processes rather than threads.
Utilizing processes enables our operating system to perform better process scheduling, mapping the process to a particular processor core on our machine (most modern operating systems are able to efficiently schedule processes that are using a lot of CPU in a parallel manner).

Running the script::

    python multiple_track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person

Improvements

- The first improvement would be to utilize processing pools rather than spawning a brand new process for each object to be tracked. If your system has N processor cores, then you would want to create a pool with N – 1 processes, leaving one core to your operating system to perform system operations.
- The second improvement would be to clean up the processes and queues.
- The third improvement would be to improve tracking accuracy by running the object detector every N frames
