from torchvision import tv_tensors
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.transforms import v2


class Dataset(Dataset):
    def __init__(self, img_files, label_files, transform=None):
        self.img_files = img_files
        self.label_files= label_files
        self.transform= transform
        
    def __len__(self):
        return len(self.img_files)

    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = self.label_files[idx]
        
        img = Image.open(img_file).convert("RGB")
        img_w, img_h= img.size
        
        
        bboxes = self.extract_bbox(label_file, img_h,img_w)
        bboxes = bboxes if bboxes else np.reshape(bboxes,(0,4))
        bboxes= tv_tensors.BoundingBoxes(bboxes,format="XYXY", canvas_size=(img_h,img_w))
        
        labels = torch.ones(bboxes.shape[0], dtype=torch.int64)
        target = {"boxes": bboxes, "labels": labels}
    
        if self.transform:
            img, target = self.transform(img, target)
            
        
        return img, target
        
    def extract_bbox(self, label_file, img_h,img_w):
        
        bboxes=[]
        with open(label_file,'r') as f:
            lines= f.readlines()
            for line in lines:
                bbox = line.split(' ')
                x_center = float(bbox[1]) * img_w
                y_center = float(bbox[2]) * img_h
                w = float(bbox[3]) * img_w
                h = float(bbox[4]) * img_h
                
                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)
                bboxes.append([x_min,y_min,x_max,y_max])
        
        return bboxes
        
    
    
    
                