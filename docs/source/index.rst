.. pycv documentation master file, created by
   sphinx-quickstart on Mon Apr 20 04:58:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyCV: Python for Computer Vision
================================

**pycv** is the collection of python projects for computer vision.

-----------

**pycv** consists of various projects based on traditional computer vision
algorithms and state-of-the-art deep learning method

- Object Tracking
- Object Detection
- Face Detection
- Face Recognition


.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation Guide
------------------

This part of the documentation covers step-by-step instructions for letting **pycv** run on your local machine.


.. toctree::
   :maxdepth: 2

   installation/python


Object Tracking
---------------

Object tracking is the task of taking an initial set of object detections, creating a unique ID for each of the initial detections, and then tracking each of the objects as they move around frames in a video, maintaining the ID assignment.


.. toctree::
   :maxdepth: 2

   object_tracking/color
   object_tracking/opencv
   object_tracking/dlib


Face Detection
--------------

Face detection is the task of detecting faces in a photo or video (and distinguishing them from other objects).


.. toctree::
   :maxdepth: 2

   face_detection/opencv_dlib
   face_detection/javascript

