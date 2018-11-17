#!/bin/idle
import glob
from shutil import copyfile
import os
emotions = ["neutral", "anger", "contempt", "disgust", "fear",
            "happy", "sadness", "surprise"]  # Define emotion order
for emotion in emotions:
    os.mkdir(os.path.join('sorted_set', emotion))
# Returns a list of all folders with participant numbers
participants = glob.glob(r"./source_emotion/*/*/*")
print(participants)
for emotion_path in participants:
    with open(emotion_path) as f:
        emotion = int(float(f.readline()))
    source_path = os.path.join('source_images', *emotion_path.split(r'/')[2:4])
    images = glob.glob(source_path + r'/*')
    source_neutral = images[0]
    source_emotion = images[-1]
    dest = '_'.join(emotion_path.split(r'/')[2:4]) + '.png'
    dest_neutral = os.path.join('sorted_set', 'neutral', dest)
    dest_emotion = os.path.join('sorted_set', emotions[emotion], dest)
    print(dest_emotion, dest_neutral)
    copyfile(source_neutral, dest_neutral)
    copyfile(source_emotion, dest_emotion)
