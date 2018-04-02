from __future__ import print_function, division
import cv2
import numpy as np
from contextlib import contextmanager
from logging import info, warning
try:
    from .utils import *
except:
    from utils import *

class HardwareError(Exception):
    """
    Custom Exception raised when External Hardware fails
    """
    pass

_imread_modes={
    'color':cv2.IMREAD_COLOR,
    'gray':cv2.IMREAD_GRAYSCALE,
    'alpha':-1
    }

def imread(img_name,mode='color'):
    img=cv2.imread(img_name,_imread_modes[mode])
    if img is None:
        raise IOError(img_name)
    return img

imwrite=cv2.imwrite

@contextmanager
def window(*args,**kwargs):
    cv2.namedWindow( *args,**kwargs )
    yield
    destroy_window(*args,**kwargs)

def destroy_window(*args,**kwargs):
    cv2.destroyWindow(*args,**kwargs)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)      
    
def imshow(img,window_name='image',hold=False,):
    if not hold:
        cv2.namedWindow( window_name )
    # if img.shape[0]>700:
    #     warning("Image size too large, resizing to fit")
    #     img = cv2.resize(img, (0,0), fx=700/img.shape[0], fy=700/img.shape[0])  
    # if img.shape[1]>700:
    #     warning("Image size too large, resizing to fit")        
    #     img = cv2.resize(img, (0,0), fx=700/img.shape[1], fy=700/img.shape[1])     
    cv2.imshow(window_name,img)
    key = cv2.waitKey(int(hold)) & 0xFF 
    if not hold:
        destroy_window(window_name)
    return chr(key)

class _imtool:
    """
    MATLAB like imtool with very limited functionality
    Show color values and position at a given point in an image, interactively
    some problems when resizing
    """
    def __init__(self,img):
        self.img=img
        self.pos=(0,0)
        with window( 'imtool' ):
            cv2.setMouseCallback('imtool', self.on_click)    
            font=cv2.FONT_HERSHEY_SIMPLEX
            while True:
                img=np.zeros_like(self.img)
                x,y=self.pos
                cols=self.img[y,x]
                text="%d %d: "%(y,x)+str(cols)
                cv2.putText(img,text,self.pos,font,.5, 255)
                key = imshow(cv2.bitwise_xor(img,self.img),window_name='imtool',hold=True)
                if key == 'q':
                    break
                try:
                    cv2.getWindowProperty('imtool', 0)
                except cv2.error:
                    break   
        
    def on_click(self, event, x, y, flags, param):           
        self.pos=(x,y)

def imtool(img):
    _imtool(img)

def crop(img,bbox,extend=0):
    (x,y,w,h)=bbox
    return img[y-extend:y+h+extend,x-extend:x+w+extend]

def overlay(destination,source):
    destination[:,:,:] = cv2.resize(source,(destination.shape[:-1][::-1]))

class imcrop:
    """
    MATLAB-like imcrop utility
    Drag mouse over area to select
    Lift to complete selection
    Doubleclick or close window to finish choosing the crop
    Rightclick to retry
    
    Example:
        >>> cropped_img,bounding_box = imcrop(img)  # cropped_img is the cropped part of the img
    
        >>> crp_obj=imcrop(img,'img_name')          # asks for interactive crop and returns an imcrop object
        <imcrop object at ...>
        
        >>> crp_obj.bounding_box                    # the bounding_box of the crop
        [[12, 15] , [134 , 232]]
        
        >>> img,bbox=imcrop(img,bbox)               # without interactive cropping
        
    """
    modes = {
        'standby': 0,
        'cropping': 1,
        'crop_finished': 2,
        'done_exit': 3}

    def __init__(self, img, window_name='image',bbox=None,):
        self.window_name =  window_name
        self.img=img
        if  bbox is None:
            self.bounding_box = []
            self.mode = 0
            self.crop()
        else:
            self.bounding_box=bbox

    def crop(self):
        cv2.namedWindow( self.window_name);
        cv2.setMouseCallback(self.window_name, self.on_click)
        while True:
            img2 = self.img.copy()
            if self.mode > 0:
                cv2.rectangle(img2, self.bounding_box[0], self.current_pos, (0, 255, 0), 1)
            key = imshow(img2,window_name=self.window_name,hold=True)
            try:
                cv2.getWindowProperty(self.window_name, 0)
            except cv2.error:
                break
            if self.mode == 3:
                break
        destroy_window(self.window_name)
        if len(self.bounding_box) != 2 or self.bounding_box[0][0] == self.bounding_box[1][0] or self.bounding_box[0][1] == self.bounding_box[1][1]:
            raise ValueError("Insufficient Points selected")

    def __iter__(self):
        bbox=self.bounding_box
        if bbox[0][0] > bbox[1][0]:
            bbox[1][0], bbox[0][0] = bbox[0][0], bbox[1][0]
        if bbox[0][1] > bbox[1][1]:
            bbox[1][1], bbox[0][1] = bbox[0][1], bbox[1][1]
        yield self.img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
        yield bbox

    def on_click(self, event, x, y, flags, param):
        if self.mode == 0 and event == cv2.EVENT_LBUTTONDOWN:
            self.mode = 1
            self.current_pos = (x, y)
            self.bounding_box = [(x, y)]
        elif self.mode == 1 and event == cv2.EVENT_LBUTTONUP:
            self.mode = 2
            self.bounding_box.append((x, y))
            self.current_pos = (x, y)
        elif self.mode == 1 and event == cv2.EVENT_MOUSEMOVE:
            self.current_pos = (x, y)
        elif self.mode == 2 and event == cv2.EVENT_RBUTTONDOWN:
            self.mode = 0
        elif self.mode == 2 and event == cv2.EVENT_LBUTTONDBLCLK:
            self.mode = 3

_kernel_shapes={
    'rectangle':cv2.MORPH_RECT,
    'square':   cv2.MORPH_RECT,
    'circle':   cv2.MORPH_ELLIPSE,
    'ellipse':   cv2.MORPH_ELLIPSE,
    'cross':    cv2.MORPH_CROSS
    } 


def _kernel(kernel_name,size):
    return cv2.getStructuringElement(_kernel_shapes[kernel_name],size)

def imdilate(img,kernel='circle',size=5,iterations=1):
    return cv2.dilate(img.copy(),_kernel(kernel,(size,size)),iterations = iterations)

def imerode(img,kernel='circle',size=5,iterations=1):
    return cv2.erode(img.copy(),_kernel(kernel,(size,size)),iterations = iterations)

def imopen(img,kernel='circle',size=5):
    return cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, _kernel(kernel,(size,size)))

def imclose(img,kernel='circle',size=5):
    return cv2.morphologyEx(img.copy(), cv2.MORPH_CLOSE, _kernel(kernel,(size,size)))

def centroid(contour):
    m = cv2.moments(contour)
    cx = int(m["m10"] / m["m00"]) 
    cy = int(m["m01"] / m["m00"])
    return np.array((cy,cx))     

def polylines(img,points,closed=False,color=(0,255,0),show_points=True):
    img=img.copy()
    if show_points:
        for point in points:
            point=tuple(map(int,point))
            cv2.circle(img,point, 2, (0,0,255), -1)
    pts = np.array(points, np.int32); 
    pts = pts.reshape((-1,1,2));
    return cv2.polylines(img,[pts],closed,color)
    
def rectangle(img,corner1,corner2,color=(255,255,255),linewidth=-1):
    cv2.rectangle(img,corner1,corner2,color,linewidth)

def find_shapes(img,show=True):
    shapes=defaultdict(list)
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            shape_name = "square"
        elif len(approx) == 5:
            shape_name = "pentagon"
        elif len(approx) == 6:
            shape_name = "hexagon"
        else:
            shape_name = "circle"
        shape=cv2.fillPoly(np.zeros_like(img), pts =[approx], color=(255,255,255))
        if show:
            imshow(shape,window_name=shape_name)
        shapes[shape_name].append(shape) 
    return shapes


class Color:
    @staticmethod
    def convert( img ,color_to,color_from ='bgr',):
        colorspace_name=('COLOR_'+color_from+'2'+color_to).upper()
        try:
            colorspace=getattr(cv2,colorspace_name)
        except AttributeError:
            raise ValueError("No such colorspace: %s"%color_to)
        return cv2.cvtColor(img,colorspace)

    @staticmethod
    def from_crop(img,colorspace=None,color_name='color',):
        img, bbox = try_to(imcrop, args=[img, color_name])
        return Color(img,colorspace)

    def __init__(self, img,colorspace=None):      
        self.color = np.zeros((2, 3))
        self.colorspace = colorspace
        if self.colorspace:
            img = Color.convert(img, colorspace)
        for i in range(3):
            self.color[0, i] = img[:, :, i].max()
            self.color[1, i] = img[:, :, i].min()
        
    def threshold(self, img, err=np.array([35,5,5])):
        if self.colorspace:
            img = Color.convert(img, self.colorspace)
        img=cv2.inRange(img, self.color[1, :] - err, self.color[0, :] + err)
        return img
    
    def __repr__(self):
        if not self.colorspace:
            colorspace='rgb'
        else:
            colorspace = self.colorspace
        return ("Mode :  "+ colorspace +'\n'+ str(self.color))

def im2bw(img,otsu=True,threshold=127):   
    img=im2gray(img)
    (thresh, im_bw) = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY | (cv2.THRESH_OTSU * int(otsu)))
    
    return im_bw
    
def im2gray(img):
        return Color.convert(img,'gray')

class Video:
    bbox = None
    released=False
    
    def __init__(self, source,resolution =[1280,720]):
        try:
            info("Starting VideoCapture")
            self.input_video = cv2.VideoCapture(source)
            info("VideoCapture Started")
        except:
            raise HardwareError("Video Not Connected")

        if resolution is not None or isinstance(source,str):
            self.input_video.set(3,resolution[0])
            self.input_video.set(4,resolution[1])
        self.input_video.grab()

    def release(self):
        if not self.released:
            self.input_video.release()
            self.released=True
            info("Video Capture Released")

    def set_roi(self,bbox=None):
        info("Setting Region of interest")
        if bbox is None:
            _,img=self.input_video.read()
            img,bbox=imcrop(img)
        self.bbox=bbox
        info ("Region of interest set as",bbox )

    def read(self):
        ret, img = self.input_video.read()
        if not ret:
            raise HardwareError("Camera Wire Pulled")
        if self.bbox is not None:
            img,bbox=imcrop(img,bbox=self.bbox)
        return img

    def __iter__(self):
        while True:
            yield self.read()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.release()

    def __del__(self):
        self.release()

def blend_transparent(face_img,overlay_img):
    overlay=cv2.resize(overlay_img,face_img.shape[:2])
    overlay_img = overlay[:,:,:3] 
    overlay_mask = overlay[:,:,-1]
    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    face_img[:,:,:] = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    




