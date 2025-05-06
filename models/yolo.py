import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import copy
import torchvision
from ultralytics import YOLO


class YOLOV8(nn.Module):
    def __init__(self,weights):
        yolo = YOLO(weights)
        
        

    def forward(self,x):
        return self.yolo(x)
    
    
    