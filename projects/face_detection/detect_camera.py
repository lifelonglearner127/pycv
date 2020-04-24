import argparse
import cv2
import imutils

import settings
import exceptions

from imutils.video import WebcamVideoStream
from pycv.utils import FPS
from detector import FaceDetectorFactory


# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--method", type=int,
                default=settings.DETECTION["VIOLA_JONES"]["METHOD"],
                help="path to the image")
args = vars(ap.parse_args())

# initialize face detector
try:
    detector = FaceDetectorFactory.build(args["method"])

except exceptions.ViolaJonesFaceDetectorConfError:
    print("[ERROR]: Viola-Jones Face Detector Not Properly Configured")

except exceptions.CaffeDNNFaceDetectorConfError:
    print("[ERROR]: Caffe DNN Face Detector Not Properly Configured")

else:
    # detect face in video stream
    stream = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    while True:
        (grabbed, frame) = stream.read()

        if not grabbed:
            print["[ERROR]: Camera error"]
            break

        frame = imutils.resize(frame, width=400)

        faces = detector.detect(frame)

        for ((x1, y1, x2, y2), confidence) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame, f"FPS: {fps.fps():.2f}", (10, 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2
        )
        cv2.imshow("Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps.update()
finally:
    cv2.destroyAllWindows()
