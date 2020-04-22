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

    python player_detect.py --video videos/match.mp4


Fast Multiple Object Tracking
-----------------------------