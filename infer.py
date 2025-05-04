from models import FasterRCNNModel, YOLOV8
import torchvision
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_NAME, DATA_DIR, CHECKPOINT_FILE,\
    SOURCE_PATH, RESULT_PATH
from utils import extract_files, Dataset, transform, data_splitting, get_dataloader, get_lr, get_logger
import copy
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
from PIL import Image
logger = get_logger(name="training_logger", log_file="logs/test.log")

class Predictor:
    def __init__(self):
        if MODEL_NAME == 'FasterRCNN':
            self.model= FasterRCNNModel()
            checkpoint= torch.load(CHECKPOINT_FILE)
            self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
            
    def preprocessing(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image = image.resize((600,600))
        image = transform['test'](image)
        image = image.unsqueeze(0)
        return image
            
    
    def predict(self, image_file):
        logger.info("Prepare Image")
        image = self.preprocessing(image_file)
        logger.info("Inference")
        prediction = self.model(image)[-1]
        pp_boxes = prediction['boxes'][prediction['scores']>=0.4]
        scores=prediction["scores"][prediction["scores"]>=0.4]
        nms=torchvision.ops.nms(pp_boxes,scores,iou_threshold=0.1)
        pp_boxes=pp_boxes[nms]

        boxes=pp_boxes.to('cpu').detach().cpu().numpy().astype(np.int32)
        image=cv2.imread(image_file)
        image = cv2.resize(image,(600,600))
        logger.info("Label Bounding Boxes")
        for box in boxes:
            image = cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(0,255,0), 3) 
        
        return image
        

def main():
    predictor = Predictor()
    for image_file in os.listdir(SOURCE_PATH):
        cv2.imwrite(f'{RESULT_PATH}/pred_{image_file}',predictor.predict(os.path.join(SOURCE_PATH,image_file)))
    logger.info("Done!!!")

if __name__ == '__main__':
    main()