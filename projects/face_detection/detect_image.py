import argparse
import datetime

import cv2
import imutils

import settings
import exceptions
from detector import FaceDetectorFactory


# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image")
ap.add_argument("-m", "--method", type=int,
                default=settings.DETECTION["VIOLA_JONES"]["METHOD"],
                help="path to the image")
args = vars(ap.parse_args())

# initialize face detector
try:
    detector = FaceDetectorFactory.build(args["method"])

except exceptions.ViolaJonesFaceDetectorConfError:
    print("[ERROR]: Viola-Jones Face Detector Not Properly Configured")

except exceptions.DLibCNNFaceDetectorConfError:
    print("[ERROR]: Dlib CNN Detector Not Properly Configured")

except exceptions.CaffeDNNFaceDetectorConfError:
    print("[ERROR]: Caffe DNN Face Detector Not Properly Configured")

else:
    # detect face in image
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=400)

    before = datetime.datetime.now()
    faces = detector.detect(image)
    after = datetime.datetime.now()
    print(f"[INFO]: It took {(after - before).total_seconds()} to detect")

    for ((x1, y1, x2, y2), confidence) in faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
