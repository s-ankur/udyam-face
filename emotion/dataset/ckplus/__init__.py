import cv2
import glob,os
import numpy as np

emotions =glob.glob(os.path.join("dataset/ckplus/dataset",'*'))
image_shape=(48,48,1)
import random

def load_data():
    training_data = []
    training_labels = []
    for label,emotion in enumerate(emotions):
        files = glob.glob(os.path.join(emotion,'*'))
        print("Emotion %s --- %d files"%(emotion,len(files)))
        random.shuffle(files)
        for file in files:
            image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)  
            training_data.append(image.reshape(image_shape))  
            training_labels.append(label)
    training_data=np.array(training_data)
    training_labels=np.array(training_labels)
    return training_data, training_labels
