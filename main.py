import os
from filters import *
from detectors import *



     

if  __name__ =='__main__':
    filter = PixelateFilter()
    with window('Overlay') ,Video(0,resolution=[500,500]) as video:
        for img in video:
            faces=extract_face(img)
            filter.apply(img,faces)
            if imshow(img,hold=True,window_name='Overlay')=='q':
                break
                
            


