import pandas as pd
import numpy as np


def load_data(path):
    data=pd.read_csv(path)    
    y=data.Emotion.as_matrix()
    X=data.Pixels
    x=[]
    for string in X:
        x(np.array(list(map(int,string.split()))))
    x=np.array(x)
    return x,y
