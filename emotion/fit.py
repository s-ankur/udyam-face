import cv2
import numpy as np
from models import cnn_model
from dataset import kaggle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import model_from_json
batch_size = 50
num_epochs = 1


def fit(train_faces, train_emotions, test_faces, test_emotions):
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    model.fit_generator(
        data_generator.flow(
            train_faces,
            train_emotions,
            batch_size),
        steps_per_epoch=len(train_faces) /
        batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(
            test_faces,
            test_emotions))


emotions = [
    'anger',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sadness',
    'surprise']


def fsit(images, labels, prediction_data, prediction_labels):
    for i in range(num_epochs):
        model.train(images, labels)
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = model.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1

            cnt += 1
        else:
            incorrect += 1
##            cv2.imshow("difficult\\%s_%s_.jpg" %(emotions[prediction_labels[cnt]], emotions[pred]), image)

# cv2.waitKey(0)
            cnt += 1
    print(((100 * correct) / (correct + incorrect)))
    model.save('trained_models/fish1')


def save_model(json_path):
    json_string = model.to_json()
    with open(json_path + '.json', 'w') as json_file:
        json_file.write(json_string)
    model.save_weights(json_path + '.h5')


def load_model(json_path):
    with open(json_path + '.json') as json_file:
        json_string = json_file.read()
    model = model_from_json(json_string)
    model.load_weights(json_path + '.h5')
    return model


if __name__ == "__main__":
    # model=cnn_model(kaggle.image_shape,len(kaggle.emotions))
    # model = cv2.face.FisherFaceRecognizer_create() # Eigenface Recognizer
    model = load_model('trained_models/fad')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    X, y = kaggle.load_data()
    X = X / 255
    y = to_categorical(y)
    training_data, prediction_data, training_labels, prediction_labels = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fit(training_data, training_labels, prediction_data, prediction_labels)
    # fit(X,y)
    save_model('trained_models/fad')
