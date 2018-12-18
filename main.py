import os
from filters import *
from detectors import *


def main():
    filter = pixelate_filter
    with window('Overlay'), Video(0, resolution=[500, 500]) as video:
        for img in video:
            try:
                features = extract_face(img)
                # features=extract_eyes(img)
                filter.apply(img, features)
                if imshow(img, hold=True, window_name='Overlay') == 'q':
                    break
            except:
                pass


if __name__ == '__main__':
    main()
