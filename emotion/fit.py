import cv2
import glob,os
import random
import numpy as np
import matplotlib.pyplot as plt
emotions =glob.glob(os.path.join("dataset",'*'))
fishface = cv2.face.createFisherFaceRecognizer()  
print(emotions)
data = {}

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

def fit(training_data, training_labels):
    print("Training")
    fishface.train(training_data, np.asarray(training_labels))
    fishface.save("tmp.yaml")


def evaluate( prediction_data, prediction_labels ):
    correct = 0
    incorrect = 0
    for cnt,image in enumerate(prediction_data):
        pred = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
        else:                                 
            print("Correct: %s   Answer: %s"%(emotions[prediction_labels[cnt]],emotions[pred]))
            if image is not None:
                #plt.imshow(image)
                #plt.show()
                incorrect += 1
            else:
                print("Blank")
    print("got", ((100*correct)/(correct + incorrect)), "percent correct!")

if __name__ =="__main__":
    training_data, training_labels, prediction_data, prediction_labels = load_data()
    fit(training_data, training_labels)
    evaluate( prediction_data, prediction_labels )
