Dlib Object Tracking
====================


People Counter
--------------

    python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/people_counting.mp4


Single Object Tracking
----------------------

    python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person


Multiple Object Tracking
------------------------

    python multiple_track_object_nmp.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person
    

Fast Multiple Object Tracking
-----------------------------

    python multiple_track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/race.mp4 --label person
