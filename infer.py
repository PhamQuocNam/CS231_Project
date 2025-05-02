from models import FasterRCNNModel, YOLOV8
import torchvision
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_NAME, DATA_DIR, CHECKPOINT_PATH,\
    SOURCE_PATH, RESULT_PATH
from utils import extract_files, Dataset, transform, data_splitting, get_dataloader, get_lr
import copy
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from PIL import Image

class Predictor:
    def __init__(self):
        if MODEL_NAME == 'RCNN':
            self.model= FasterRCNNModel()
            checkpoint= torch.load(CHECKPOINT_PATH)
            self.model.load_state_dict(checkpoint['model_state'])
            
    def preprocessing(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image = transform['test'](image)
        image = image.unsqueeze(0)
        return image
            
    
    def predict(self, image_file):
        image = self.preprocessing(image_file)
        prediction = self.model(image)
        pp_boxes = prediction['boxes'][prediction['scores']>=0.4]
        scores=prediction["scores"][prediction["scores"]>=0.4]
        nms=torchvision.ops.nms(pp_boxes,scores,iou_threshold=0.1)
        pp_boxes=pp_boxes[nms]
        
        boxes=pp_boxes.to('cpu').detach().cpu().numpy().astype(np.int32)
        image=image.to('cpu').permute(1,2,0).numpy().copy()    
        image = (image * 255).clip(0, 255).astype(np.uint8)
        for box in boxes:
            image = cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(0,255,0), 3) 
        
        return image
        


def main():
    predictor = Predictor()
    for image_file in os.listdir(SOURCE_PATH):
        cv2.imwrite(f'{RESULT_PATH}/pred_{image_file}',predictor.predict(image_file))
    

if __name__ == '__main__':
    main()