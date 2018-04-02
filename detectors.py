from utility.cv_utils import *
import os

_detector_names=(
"haarcascade_frontalface_default.xml",
"haarcascade_frontalface_alt2.xml",
"haarcascade_frontalface_alt.xml",
"haarcascade_frontalface_alt_tree.xml")

_detectors=[ cv2.CascadeClassifier(os.path.join('models', detector_name)) for detector_name in _detector_names]

def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for detector in _detectors:
        features = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(5, 5),
            flags=cv2.CASCADE_SCALE_IMAGE)
        if len( features ) :break
    return features
