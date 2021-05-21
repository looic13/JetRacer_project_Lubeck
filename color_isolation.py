import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def colorIsolationTransform(path,color):
    img=cv2.imread(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if color=='orange':
        lower = np.array([0, 50,100])
        upper = np.array([60,255, 255])
    if color=='red':
        lower = np.array([170, 100,100])
        upper = np.array([180, 255, 255])
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange)")
        
        
    mask = cv2.inRange(hsv, lower, upper)
    mask[0:90,0:224]=0
    
    return mask
    
    
def colorIsolationPreprocess(img,color):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if color=='orange':
        lower = np.array([0, 50,100])
        upper = np.array([60,255, 255])
    if color=='red':
        lower = np.array([170, 100,100])
        upper = np.array([180, 255, 255])
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange)")
        
        
    mask = cv2.inRange(hsv, lower, upper)
    mask[0:90,0:224]=0
    
    return mask
    
    