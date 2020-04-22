import argparse
import cv2
import dlib
import imutils
import multiprocessing as mp
import numpy as np
from imutils.video import FPS


def start_tracker(box, label, rgb, input_queue, output_queue):
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    tracker.start_track(rgb, rect)

    while True:
        rgb = input_queue.get()
        if rgb is not None:
            tracker.update(rgb)
            pos = tracker.get_position()
            start_x = int(pos.left())
            start_y = int(pos.top())
            end_x = int(pos.right())
            end_y = int(pos.bottom())
            output_queue.put((label, (start_x, start_y, end_x, end_y)))


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-l", "--label", required=True,
                help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


input_queue = []
output_queue = []

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"])
trackers = []
writer = None
label = ""

fps = FPS().start()

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    if len(input_queue) == 0:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            label = CLASSES[int(detections[0, 0, i, 1])]
            if conf > args["confidence"] and label == args["label"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                iq = mp.Queue()
                oq = mp.Queue()
                input_queue.append(iq)
                output_queue.append(oq)

                p = mp.Process(
                    target=start_tracker, args=(
                        (start_x, start_y, end_x, end_y), label, rgb, iq, oq
                    )
                )
                p.daemon = True
                p.start()

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (start_x, start_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    else:
        for iq in input_queue:
            iq.put(rgb)

        for oq in output_queue:
            label, (start_x, start_y, end_x, end_y) = oq.get()
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                          (0, 255, 0), 2)
            cv2.putText(frame, "person", (start_x, start_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if writer is not None:
        writer.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()

vs.release()
cv2.destroyAllWindows()
