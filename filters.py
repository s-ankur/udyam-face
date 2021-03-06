import numpy as np
from emotion.dataset import kaggle
from keras.models import model_from_json
import cv2
from utility.cv_utils import *
import glob
import os
from detectors import extract_eyes


class Filter:
    def apply(self, img, faces):
        for face in faces:
            self(img, face)


class OverlayFilter(Filter):
    overlay_img = imread(os.path.join('data', "pink_flower.png"), mode='alpha')

    def __init__(self, img=None, extend=10):
        if img is not None:
            self.overlay_img = img
        self.extend = extend

    def __call__(self, image, bbox):
        face_img = crop(image, bbox, self.extend)
        blend_transparent(face_img, self.overlay_img)


class BlurFilter(Filter):
    def __call__(self, image, bbox):
        face_img = crop(image, bbox, extend=10)
        mask = im2bw(face_img, otsu=True, threshold=10)
        mask = (mask == 0)
        try:
            mask3d = np.zeros(mask.shape + (3,))
            mask3d[:, :, 0] = mask
            mask3d[:, :, 1] = mask
            mask3d[:, :, 2] = mask
            face_img[:, :, :] = np.where(
                mask3d, cv2.blur(face_img, (10, 10)), face_img)

        except BaseException:
            pass


class PixelateFilter(Filter):
    def __init__(self, pixellation=.09):
        self.pixellation = pixellation

    def __call__(self, image, bbox):
        face_img = crop(image, bbox, extend=10)
        mask = im2bw(face_img, otsu=True, threshold=10)
        mask = (mask == 0)

        mask3d = np.zeros(mask.shape + (3,))
        mask3d[:, :, 0] = mask
        mask3d[:, :, 1] = mask
        mask3d[:, :, 2] = mask

        new_img = cv2.resize(
            face_img,
            (0,
             0),
            fx=self.pixellation,
            fy=self.pixellation,
            interpolation=cv2.INTER_LINEAR)
        new_img = cv2.resize(new_img,
                             face_img.shape[:-1][::-1],
                             fx=0,
                             fy=0,
                             interpolation=cv2.INTER_NEAREST)
        face_img[:, :, :] = np.where(mask3d, new_img, face_img)


class FFilter(Filter):

    def __call__(self, image, bbox):
        face_img = crop(image, bbox, extend=5)
        mask = im2bw(face_img, otsu=True, threshold=10)
        mask = (mask == 0)
        try:
            mask3d = np.zeros(mask.shape + (3,))
            mask3d[:, :, 0] = mask
            mask3d[:, :, 1] = mask
            mask3d[:, :, 2] = mask

            new_img = 255 - face_img
            face_img[:, :, :] = np.where(mask3d, new_img, face_img)
        except BaseException:
            pass


class PixelateFilter(Filter):
    def __init__(self, pixellation=.09):
        self.pixellation = pixellation

    def __call__(self, image, bbox):
        face_img = crop(image, bbox, extend=10)
        mask = im2bw(face_img, otsu=True, threshold=10)
        mask = (mask == 0)

        mask3d = np.zeros(mask.shape + (3,))
        mask3d[:, :, 0] = mask
        mask3d[:, :, 1] = mask
        mask3d[:, :, 2] = mask

        new_img = cv2.resize(
            face_img,
            (0,
             0),
            fx=self.pixellation,
            fy=self.pixellation,
            interpolation=cv2.INTER_LINEAR)
        new_img = cv2.resize(new_img,
                             face_img.shape[:-1][::-1],
                             fx=0,
                             fy=0,
                             interpolation=cv2.INTER_NEAREST)
        face_img[:, :, :] = np.where(mask3d, new_img, face_img)


class VideoFilter(Filter):
    frame = 0

    def __init__(self, image_dir):
        self.images = [imread(image_path, mode='alpha') for image_path in
                       glob.glob(os.path.join('data', image_dir, '*'))]

    def __call__(self, image, bbox, *args):
        face_img = crop(image, bbox)
        if self.frame >= len(self.images):
            self.frame = 0
        blend_transparent(face_img, self.images[self.frame])
        self.frame += 1


def blend_overlay(face_img, new_img):
    mask = im2bw(face_img, otsu=True, threshold=10)
    mask = (mask == 0)
    mask3d = np.zeros(mask.shape + (3,))
    mask3d[:, :, 0] = mask
    mask3d[:, :, 1] = mask
    mask3d[:, :, 2] = mask
    new_img = cv2.resize(new_img, face_img.shape[:2])
    imshow(mask3d, hold=True)
    face_img[:, :, :] = np.where(mask3d, new_img, face_img)


class SwapFaces:
    def apply(self, image, bboxes):
        faces = [crop(image, bbox) for bbox in bboxes]
        old_face = None
        for face in faces:
            tmp = face.copy()
            if old_face is not None:
                overlay(face, old_face)
            old_face = tmp
        if len(faces) > 1:
            overlay(faces[0], old_face)


#from emotion.dataset import kaggle


class EmotionFilter(Filter):
    # emotions=['anger','disgust','fear','happy','neutral','sadness','surprise']
    emotions = kaggle.emotions
    emotion_pics = [
        imread(
            r'data/' +
            emotion +
            '.png',
            mode='alpha') for emotion in kaggle.emotions]
    color = (255, 255, 0)
    colors = ((255, 0, 0), (100, 100, 50), (255, 255, 0),
              (0, 255, 0), (0, 0, 255), (255, 255, 255), (50, 50, 50))
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, json_path):
        with open(json_path + '.json') as json_file:
            json_string = json_file.read()
        self.model = model_from_json(json_string)
        self.model.load_weights(json_path + '.h5')

    def draw_emotions(self, emotion_v, img):
        i = 0
        for emotion, val in zip(self.emotions, emotion_v):
            a = np.array([15, val * 30])
            bbox = np.array([15 * i, 0])
            cv2.rectangle(img, tuple(bbox[:2].astype('int')), tuple(
                (bbox[:2] + a).astype('int')), self.colors[i], -1)
            i += 1

    def __call__(self, img, bbox):
        face = crop(img, bbox)
        eyes = extract_eyes(face)
        face_img = Color.convert(face, 'gray')
        face_img = cv2.resize(face_img, kaggle.image_shape[:2])
        emotion_v = self.model.predict(
            (face_img / 255).reshape(1, *kaggle.image_shape))
        emotion_v[0, 1] = 0
        emotion_id = emotion_v[0, :].argmax(0)
        for bbox_eye in eyes:
            eye = crop(face, bbox_eye)
            blend_transparent(eye, self.emotion_pics[emotion_id])
        #cv2.rectangle(img,tuple( bbox[:2]),tuple(bbox[:2]+ bbox[2:]), self.color, 3)
        cv2.putText(img, self.emotions[emotion_id], tuple(
            bbox[:2]), self.font, .5, 255)
        self.draw_emotions(emotion_v[0, :], img)


flame = imread(r'data/flame.png', mode='alpha')
much = imread(r'data/much.png', mode='alpha')
specs = imread(r'data/specs.png', mode='alpha')
muck = imread(r'data/muck.png', mode='alpha')
eye_filter = OverlayFilter(flame)
video_filter = VideoFilter('flame')
pixelate_filter = PixelateFilter()
moustace_filter = OverlayFilter(much)
specs_filter = OverlayFilter(specs)
ffilter = FFilter()
muck_filter = OverlayFilter(muck, 50)
emotion_filter = EmotionFilter(r'emotion/trained_models/fad')
