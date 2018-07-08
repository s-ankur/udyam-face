emotions = ('Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')


def load_data(name):
    import pandas as pd
    import numpy as np
    path = name+'.csv'
    data = pd.read_csv(path)
    y = data.Emotion.as_matrix()
    x = []
    for string in data.Pixels:
        x.append(np.array(list(map(int, string.split()))))
    x = np.array(x)
    return x, y
