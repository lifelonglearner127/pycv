import os

import cv2
import dlib
import imutils
import numpy as np

import exceptions
import settings


class CascadeFaceDetector:
    """Viola-Jones face detector"""

    def __init__(self):
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)

        if not os.path.exists(
            settings.DETECTION["VIOLA_JONES"]["CASCADE_FILE"]
        ):
            raise exceptions.CascadeFileDoesNotExist()

        self.detector = cv2.CascadeClassifier(
            settings.DETECTION["VIOLA_JONES"]["CASCADE_FILE"]
        )

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size
        )
        faces = [((x, y, x + w, y + h), 1) for (x, y, w, h) in faces]
        return faces

    def __repr__(self):
        return "CascadeFaceDetector"

    def __str__(self):
        return "CascadeFaceDetector"


class DlibHOGFaceDetector:
    """HOG + SVM face detector"""

    def __init__(self):
        # number_of_times_to_upsample: How many times to upsample the image
        # looking for faces. Higher numbers find smaller faces.
        self.number_of_times_to_upsample = 0
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, self.number_of_times_to_upsample)
        faces = [
            ((face.left(), face.top(), face.right(), face.bottom()), 1)
            for face in faces
        ]
        return faces

    def __repr__(self):
        return "HOGFaceDetector"

    def __str__(self):
        return "HOGFaceDetector"


class DlibCNNFaceDetector:
    """Maximum-Margin Object Detector with CNN"""

    def __init__(self):
        try:
            self.number_of_times_to_upsample = 2
            self.detector = dlib.cnn_face_detection_model_v1(
                settings.DETECTION["DLIB_CNN"]["MODE_FILE"]
            )
        except Exception:
            raise exceptions.DLibCNNFaceDetectorConfError

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, self.number_of_times_to_upsample)
        faces = [(
            (
                face.rect.left(), face.rect.top(),
                face.rect.right(), face.rect.bottom()
            ), 1)
            for face in faces
        ]
        return faces

    def __repr__(self):
        return "HOGFaceDetector"

    def __str__(self):
        return "HOGFaceDetector"


class CaffeFaceDetector:
    """Caffe model dnn face detector"""

    def __init__(self):

        try:
            self.r_mean = settings.DETECTION["CAFFE"]["R_MEAN"]
            self.g_mean = settings.DETECTION["CAFFE"]["G_MEAN"]
            self.b_mean = settings.DETECTION["CAFFE"]["B_MEAN"]
            self.confidence = settings.DETECTION["CAFFE"]["CONFIDENCE"]
        except Exception:
            raise exceptions.CaffeModelConfigurationError()

        if not os.path.exists(settings.DETECTION["CAFFE"]["CONFIG_FILE"]):
            raise exceptions.CaffeConfigFileDoesNotExist()

        if not os.path.exists(settings.DETECTION["CAFFE"]["MODEL_FILE"]):
            raise exceptions.CaffeModelFileDoesNotExist()

        self.detector = cv2.dnn.readNetFromCaffe(
            settings.DETECTION["CAFFE"]["CONFIG_FILE"],
            settings.DETECTION["CAFFE"]["MODEL_FILE"]
        )

    def detect(self, image):
        faces = []
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            imutils.resize(image, 300, 300), scalefactor=1.0,
            mean=(self.r_mean, self.g_mean, self.b_mean), swapRB=True
        )
        self.detector.setInput(blob)
        face_candidates = self.detector.forward()

        for i in range(0, face_candidates.shape[2]):
            confidence = face_candidates[0, 0, i, 2]
            if confidence < self.confidence:
                continue

            face_rect = face_candidates[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = face_rect.astype("int")
            faces.append(((x1, y1, x2, y2), confidence))

        return faces

    def __repr__(self):
        return "CaffeFaceDetector"

    def __str__(self):
        return "CaffeFaceDetector"


class FaceDetectorFactory:

    @classmethod
    def build(self, method):
        detections = {
            settings.DETECTION["VIOLA_JONES"]["METHOD"]: CascadeFaceDetector,
            settings.DETECTION["DLIB_HOG"]["METHOD"]: DlibHOGFaceDetector,
            settings.DETECTION["DLIB_CNN"]["METHOD"]: DlibCNNFaceDetector,
            settings.DETECTION["CAFFE"]["METHOD"]: CaffeFaceDetector,
        }
        return detections[method]()
