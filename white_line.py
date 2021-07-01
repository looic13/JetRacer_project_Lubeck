import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import time

def lane_islonger(mask):
    liste=[]
    coffre=[]
    max_len=0
    twolines=False

    histogram = np.sum(mask[90:224,0:224], axis=0)
    for indice in range(0,len(histogram)):
        if histogram[indice]>0:
            coffre.append(indice)
        if (len(coffre)!=0 and len(coffre)>10 and histogram[indice]==0) or (len(coffre)>10 and indice==len(histogram)-1) :
            liste.append(coffre)
            coffre=[]

    if len(liste)>1:
        twolines=True
        for nbliste in range(0,len(liste)-1):
            if len(liste[nbliste])>len(liste[nbliste+1]):
                max_len=nbliste
            else:
                max_len=nbliste+1

    if np.mean(liste[max_len])>112:
        return 'right',twolines#,liste[max_len][-1]+1
    else :
        return 'left',twolines#,liste[max_len][0]+1

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def translate_image(image,p_width,p_height):
    height, width = image.shape[:2]
    transl_mat = np.float32([[1, 0, p_width], [0, 1, p_height]])
    result = cv2.warpAffine(image, transl_mat, (width, height))
    return result

def draw_centerline_Transform(path):
    img=cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask= cv2.inRange(hsv, (90, 95,0), (130,255, 255))
    mask[0:90,0:224]=0
    a,b = lane_islonger(mask)
    thinned = cv2.ximgproc.thinning(mask)
    #a = longest line ; b = twolines true or false ; ####c = start line
    if a == 'right':
        thinned=rotate_image(thinned,-25)
        thinned=translate_image(thinned,-60,20)
    else: 
        thinned=rotate_image(thinned,25)
        thinned=translate_image(thinned,60,20)
        
    return mask+thinned

def draw_centerline_Preprocess(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask= cv2.inRange(hsv, (90, 95,0), (130,255, 255))
    mask[0:90,0:224]=0
    a,b = lane_islonger(mask)
    thinned = cv2.ximgproc.thinning(mask)
    #a = longest line ; b = twolines true or false ; ####c = start line
    if a == 'right':
        thinned=rotate_image(thinned,-25)
        thinned=translate_image(thinned,-60,20)
    else: 
        thinned=rotate_image(thinned,25)
        thinned=translate_image(thinned,60,20)
        
    return mask+thinned

def mask_canny_Transform(path):
    img=cv2.imread(path)
    blur_img = cv2.medianBlur(img, 15)
    gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,30, apertureSize=3)
    edges[0:125,0:224]=0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask= cv2.inRange(hsv, (90, 98,0), (130,255, 255))
    mask+=edges

    mask[0:90,0:224]=0
    
    return mask

def mask_canny_Preprocess(img):
    blur_img = cv2.medianBlur(img, 15)
    gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,30, apertureSize=3)
    edges[0:125,0:224]=0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask= cv2.inRange(hsv, (90, 98,0), (130,255, 255))
    mask+=edges

    mask[0:90,0:224]=0
    
    return mask