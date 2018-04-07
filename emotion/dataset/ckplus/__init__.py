import cv2
import glob,os
import random
emotions =glob.glob(os.path.join("dataset",'*'))



def load_data():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for label,emotion in enumerate(emotions):
        files = glob.glob(os.path.join(emotion,'*'))
        print("Emotion %s --- %d files"%(emotion,len(files)))
        random.shuffle(files)
        training = files[:int(len(files)*0.8)]  
        prediction = files[-int(len(files)*0.2):]  
        for item in training:
            image = cv2.imread(item)  
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            training_data.append(gray)  
            training_labels.append(label)
        for item in prediction:
            image = cv2.imread(item)    
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(label)
    return training_data, training_labels, prediction_data, prediction_labels
