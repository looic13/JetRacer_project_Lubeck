import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time


def colorIsolationTransform(path,color):
    t0=time.time()
    img=cv2.imread(path)
    
    if color=='orange':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
            upper1 = np.array([180, 255, 100])
        
            lower2 = np.array([0,100,0])
            upper2 = np.array([5,255,100])
            
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
    
    elif color=='orangeBGR':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask= cv2.inRange(hsv, (90, 100,0), (130,255, 255))
        mask[0:90,0:224]=0
       
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange or orangeBGR)")
    
    t1=time.time()    
    temps=t1-t0
    return mask,temps
    
    
def colorIsolationPreprocess(img,color):
    
    if color=='orange':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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

    elif color=='orangeBGR':
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask= cv2.inRange(hsv, (90, 100,0), (130,255, 255))
        mask[0:90,0:224]=0
        
    else: 
        sys.exit("Error 2nd argument: colorIsolation(path,red or orange or orangeBGR)")
    
    return mask
    
def colorIsolationPreprocess2(img,color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color=='red':
        lower1 = np.array([170, 100,100])
        upper1 = np.array([180, 255, 255])
        
        lower2 = np.array([0,100,100])
        upper2 = np.array([5,255,255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 =cv2.inRange(hsv, lower2, upper2)
        mask_first=mask1+mask2
        mask_first[0:90,0:224]=0
        
    return mask_first

def colorIsolationTestImages(path):
    
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
    
    img=cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    imgbw=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #orange line isolation
    mask1= cv2.inRange(hsv, (90, 100,0), (130,255, 255))

    #White line isolation
    #thresh_bin= cv2.adaptiveThreshold(imgbw, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    ret,thresh_bin2=cv2.threshold(imgbw,140,255,cv2.THRESH_BINARY)
    
 
    #edge detection
    sobelx = cv2.Sobel(imgbw,cv2.CV_64F,1,0,ksize=5)
    abs_sobelx=np.absolute(sobelx)
    ret,thresh_bin=cv2.threshold(abs_sobelx,190,255,cv2.THRESH_BINARY)
    thresh_bin[0:90,0:224]=0
    
    #canny
    edges=cv2.GaussianBlur(imgbw,(3,3),cv2.BORDER_DEFAULT)
    edges = cv2.Canny(edges,10,30)
    edges[0:90,0:224]=0
    
    #inrange
    imgbwinrange=cv2.inRange(imgbw,130,135)
    
    #addition
    mask=mask1 #+thresh_bin
    mask[0:90,0:224]=0
    
    #TEST ADAPTATIVE THRESHOLD
    histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
    max1=np.max(histogram[0:112])
    max2=np.max(histogram[112:224])

    if max1>max2:
        midpoint=np.argmax(histogram[0:112]==max1)+120
    else :
        midpoint=np.argmax(histogram[112:224]==max2)-120 

    whitemean= np.mean(imgbw[125:,midpoint-10:midpoint+10])
    maskwhite=cv2.inRange(imgbw,whitemean-5,whitemean+5)
    
    #THINNING method
    thinned = cv2.ximgproc.thinning(mask)
    thinned[0:200,70:170]=thinned[0:200,0:100]
    thinned[0:224,0:100]=0
    thinnedrotate=rotate_image(thinned,30)
    translateimage= translate_image(thinnedrotate,0,40)
    total=mask+translateimage

    #plot
    plt.figure(1)
    plt.subplot(171)
    plt.imshow(img)
    plt.title("normal")
    plt.axis('off')
    
    plt.subplot(172)
    plt.imshow(mask1,cmap='gray')
    plt.title("orange")
    plt.axis('off')

    plt.subplot(173)
    plt.imshow(thresh_bin,cmap='gray')
    plt.title("sobel")
    plt.axis('off')

    plt.subplot(174)
    plt.imshow(edges,cmap='gray')
    plt.title("canny")
    plt.axis('off')
    
    plt.subplot(175)
    plt.imshow(thresh_bin2,cmap='gray')
    plt.title("binary")
    plt.axis('off')
    
    plt.subplot(176)
    plt.imshow(imgbwinrange,cmap='gray')
    plt.title("inrange")
    plt.axis('off')
    
    plt.subplot(177)
    plt.imshow(total,cmap='gray')
    plt.title('total')
    plt.axis('off')
    
    plt.show()
    
    