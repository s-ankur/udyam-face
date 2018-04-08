
import glob
from shutil import copyfile
import os
def makedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
makedir('sorted_set')
for emotion in emotions :
    makedir(os.path.join('sorted_set',emotion))
participants = glob.glob(os.path.join(r"source_emotion",'*','*',"*")) #Returns a list of all folders with participant numbers
#print(participants)
for emotion_path in participants:
    with open(emotion_path) as f:
            emotion = int(float(f.readline()))
    source_path= os.path.join('source_images',* emotion_path.split('\\')[1:3])
    print(source_path)
    images= glob.glob(os.path.join(source_path,'*'))
    source_neutral=images[0]
    source_emotion=images[-1]
    dest='_'.join(emotion_path.split('\\')[1:3])+'.png'
    dest_neutral=os.path.join('sorted_set','neutral',dest)
    dest_emotion=os.path.join('sorted_set',emotions[emotion],dest)
    print(dest_emotion,dest_neutral)
    copyfile(source_neutral, dest_neutral) 
    copyfile(source_emotion, dest_emotion)
