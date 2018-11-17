import pandas as pd
import numpy as np

image_shape = (48, 48, 1)
emotions = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def load_data(path='dataset/kaggle/train.csv'):
    data = pd.read_csv(path)
    y = data.Emotion.as_matrix()
    X = data.Pixels
    x = []
    for string in X:
        x.append(np.array(list(map(int, string.split()))).reshape(image_shape))
    x = np.array(x)
    return x, y
