import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time

def colorIsolationTransform(path,color):
    t0=time.time()
    img=cv2.imread(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if color=='orange':
        lower = np.array([0, 50,100])
        upper = np.array([60,255, 255])
        mask_first = cv2.inRange(hsv, lower, upper)
        mask_first[0:90,0:224]=0
        percentage1= np.sum(mask_first==0)/(224*224)
        
        if percentage1>0.95 or percentage1<0.89:
            lower = np.array([0, 50,0])
            upper = np.array([60,255, 120])
            mask_second = cv2.inRange(hsv, lower, upper)
            mask_second[0:90,0:224]=0
            percentage2= np.sum(mask_second==0)/(224*224)
            
            if (percentage2>0.95 or percentage2<0.89) & (percentage1 < percentage2) :
                mask=mask_first
            else:
                mask=mask_second
        else : 
            mask = mask_first

            
        
    elif color=='red':
        lower1 = np.array([170, 100,100])
        upper1 = np.array([180, 255, 255])
        
        lower2 = np.array([0,100,100])
        upper2 = np.array([5,255,255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 =cv2.inRange(hsv, lower2, upper2)
        mask_first=mask1+mask2
        mask_first[0:90,0:224]=0
        percentage1=np.sum(mask_first==0)/(224*224)
        
        if percentage1>0.95 or percentage1<0.89:
            lower1 = np.array([170, 100,0])
            upper1 = np.array([180, 255,120 ])

            lower2 = np.array([0,100,0])
            upper2 = np.array([5,255,120])

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 =cv2.inRange(hsv, lower2, upper2)
            mask_second=mask1+mask2
            mask_second[0:90,0:224]=0
            percentage2= np.sum(mask_second==0)/(224*224)
            
            if (percentage2>0.95 or percentage2<0.89) & (percentage1 < percentage2) :
                mask=mask_first
            else:
                mask=mask_second
        else : 
            mask = mask_first

       
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange)")
    
    t1=time.time()    
    temps=t1-t0
    return mask,temps
    
    
def colorIsolationPreprocess(img,color):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if color=='orange':
        lower = np.array([0, 50,100])
        upper = np.array([60,255, 255])
        mask_first = cv2.inRange(hsv, lower, upper)
        mask_first[0:90,0:224]=0
        percentage1= np.sum(mask_first==0)/(224*224)
        
        if percentage1>0.95 or percentage1<0.89:
            lower = np.array([0, 50,0])
            upper = np.array([60,255, 120])
            mask_second = cv2.inRange(hsv, lower, upper)
            mask_second[0:90,0:224]=0
            percentage2= np.sum(mask_second==0)/(224*224)
            
            if (percentage2>0.95 or percentage2<0.89) & (percentage1 < percentage2) :
                mask=mask_first
            else:
                mask=mask_second
        else : 
            mask = mask_first

            
        
    elif color=='red':
        lower1 = np.array([170, 100,100])
        upper1 = np.array([180, 255, 255])
        
        lower2 = np.array([0,100,100])
        upper2 = np.array([5,255,255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 =cv2.inRange(hsv, lower2, upper2)
        mask_first=mask1+mask2
        mask_first[0:90,0:224]=0
        percentage1=np.sum(mask_first==0)/(224*224)
        
        if percentage1>0.95 or percentage1<0.89:
            lower1 = np.array([170, 100,0])
            upper1 = np.array([180, 255,120 ])

            lower2 = np.array([0,100,0])
            upper2 = np.array([5,255,120])

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 =cv2.inRange(hsv, lower2, upper2)
            mask_second=mask1+mask2
            mask_second[0:90,0:224]=0
            percentage2= np.sum(mask_second==0)/(224*224)
            
            if (percentage2>0.95 or percentage2<0.89) & (percentage1 < percentage2) :
                mask=mask_first
            else:
                mask=mask_second
        else : 
            mask = mask_first

       
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange)")
    
    return mask
    
    