import os
from  filters import *
from detectors import *



if  __name__ =='__main__':
    filter = muck_filter
    with window('Overlay') ,Video(0,resolution=[500,500]) as video:
        for img in video:
            features=extract_face(img)
            #features=extract_eyes(img)
            try:
                filter.apply(img,features)
            except :
                pass
            if imshow(img,hold=True,window_name='Overlay')=='q':
                break
                
            


