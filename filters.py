import cv2
from utility.cv_utils import *
import glob,os

class Filter:
    def apply(self,img,faces):
        for face in (faces): 
            self(img,face)

class OverlayFilter(Filter):
    overlay_img=imread(os.path.join('dataset',"pink_flower.png"),mode='alpha')

    def __call__(self,image, bbox):
        face_img=crop(image,bbox)
        blend_transparent(face_img,self.overlay_img)

class BlurFilter(Filter):
    def __call__(self,image, bbox):
        face_img=crop(image,bbox,extend=10)
        mask=im2bw(face_img,otsu=True,threshold=10)
        mask=(mask==0)
        
        mask3d=np.zeros(mask.shape+(3,))
        mask3d[:,:,0]=mask
        mask3d[:,:,1]=mask
        mask3d[:,:,2]=mask
        face_img[:,:,:] =np.where(mask3d, cv2.blur(face_img,(10,10)),face_img)

class PixelateFilter(Filter):
    def __init__(self,pixellation=.09):
        self.pixellation=pixellation

    def __call__(self,image, bbox):
        face_img=crop(image,bbox,extend=10)
        mask=im2bw(face_img,otsu=True,threshold=10)
        mask=(mask==0)
        
        mask3d=np.zeros(mask.shape+(3,))
        mask3d[:,:,0]=mask
        mask3d[:,:,1]=mask
        mask3d[:,:,2]=mask

        new_img=cv2.resize(face_img,(0,0),fx=self.pixellation, fy=self.pixellation, interpolation = cv2.INTER_LINEAR )
        new_img=cv2.resize(new_img,face_img.shape[:-1][::-1],fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        face_img[:,:,:] =np.where(mask3d,new_img,face_img)



class VideoFilter(Filter):
    frame=0

    def __init__(self,image_dir):
        self.images = [imread(image_path,mode='alpha') for image_path in glob.glob(os.path.join('dataset',image_dir,'*'))]

    def __call__(self,image,bbox,*args):
        face_img=crop(image,bbox)
        if self.frame>=len(self.images):
            self.frame=0
        blend_transparent(face_img,self.images[self.frame])
        self.frame+=1

class SwapFaces:
    def apply(self,image,bboxes):
        faces= [crop(image,bbox) for bbox in bboxes ]
        old_face=None
        for face in faces:
            tmp=face.copy()
            if old_face is not None:
                overlay(face,old_face)
            old_face=tmp
        if len(faces)>1:
            overlay(faces[0],old_face)
            

    
        
        
