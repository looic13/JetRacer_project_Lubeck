import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
from color_isolation import colorIsolationPreprocess2
#from meanstd import get_meanstd
#meann,stdd=get_meanstd('road_following_orange_line')
#mean=meann.cuda()
#std=stdd.cuda()
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    #image=colorIsolationPreprocess2(image,'red')
    image = PIL.Image.fromarray(image)
    #image=transforms.functional.to_grayscale(image,num_output_channels=1)
    image = transforms.functional.to_tensor(image).to(device)
    #image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]