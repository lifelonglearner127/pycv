OpenCV Object Tracking
======================


Centroid Tracking Algorithm
---------------------------
Algorithm:

- Accept bounding box coordinates for each object in every frame
- Compute the Euclidean distance between the centroids of the input bounding boxes and the centroids of existing objects that we already have examined.
- Update the tracked object centroids to their new centroid locations based on the new centroid with the smallest Euclidean distance.
- Register new objects
- Deregister old objects

Limitations and drawbacks:

- It requires that we run an object detector for each frame of the video — if your object detector is computationally expensive to run you would not want to utilize this method.
- It does not handle overlapping objects well and due to the nature of the Euclidean distance between centroids, it’s actually possible for our centroids to “swap IDs” which is far from ideal.

Running the script::

    python centroid_tracker.py --video videos/ball_tracking.mp4 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel


Single Object Trackiing
-----------------------
OpenCV include 8 object tracking algorithms.

- **CSRT**
- **KCF**
- **Boosting**
- **MIL**
- **TLD**
- **MedianFlow**
- **MOSSE**
- **GOTURN**

Running the script::

    python opencv_tracker.py --video videos/ball_tracking.mp4


Multiple Object Tracking
------------------------
This example is a simple extension of the above example. This example let you tracks multiple objects at the same time. But it might decrease the by the number of trackers.

Running the script::

    python multiple_opencv_tracker.py --video videos/ball_movement.mp4

