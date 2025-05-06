from .utils.logger import get_logger
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from .models import FasterRCNNModel, YOLOV8
import torchvision
from .config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_NAME, DATA_DIR, CHECKPOINT_PATH
from .utils import extract_files, Dataset, transform, data_splitting, get_dataloader, get_lr, save_checkpoint, check
import copy
logger = get_logger(name="training_logger", log_file="logs/train.log")


class Trainer:
    def __init__(self):
        if MODEL_NAME == 'FasterRCNN':
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = FasterRCNNModel(weights).to(DEVICE)
        else:
            logger.error("MODEL_NAME is not suitable!")
            raise Exception("MODEL_NAME is not suitable!")
        
        data_dir = DATA_DIR
        try:
            data_files = os.listdir(data_dir)
        except:
            logger.error("DATA_DIR is not available!!!")
            FileNotFoundError("DATA_DIR is not available!")
        
        total_image_files, total_label_files= [],[]
        
        for file in data_files:
            image_files, label_files= extract_files(file)
            total_image_files += image_files
            total_label_files += label_files
            
        train_image_files,val_image_files, train_label_files, val_label_files= data_splitting(total_image_files, total_label_files)
        train_dataset = Dataset(train_image_files, train_label_files,transform['train'])
        val_dataset =  Dataset(val_image_files, val_label_files, transform['val'])
        self.train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE,  weight_decay=0.1)
        self.lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, 
                                                  patience=8, threshold=0.0001)
            
            
        
    def run(self):
        self.best_validation_loss= np.inf
        self.best_weights= self.model.state_dict() 
        self._train()
        save_checkpoint(MODEL_NAME, self.best_weights, self.optimizer, EPOCHS,CHECKPOINT_PATH )
        logger.info("Done!!!")
  
    
    def _train(self):
        total_training_loss= []
        total_valid_loss=[]
        self.model.train()
        
        for epoch in range(EPOCHS):
            epoch_loss=[]
            current_lr=get_lr(self.optimizer)
            for imgs, targets in self.train_loader:
                self.optimizer.zero_grad()
                imgs = list(img.to(DEVICE) for img in imgs)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(imgs, targets)
                losses=sum(loss for loss in loss_dict.values())
                epoch_loss.append(losses.item())
                losses.backward()    
                self.optimizer.step()
            epoch_loss=sum(epoch_loss)/len(epoch_loss)  
            total_training_loss.append(epoch_loss)
            
            val_loss= self.eval(current_lr)
            total_valid_loss.append(val_loss)
            logger.info(f"Epoch {epoch+1} \nTraining Loss: {epoch_loss} \nValidation Loss: {val_loss}")
        
        logger.info(f"Average Training: {sum(total_training_loss)/len(total_training_loss)}")
        logger.info(f"Average Validation: {sum(total_valid_loss)/len(total_valid_loss)}")
        
        
    def eval(self, current_lr ):
        total_loss=[]
        with torch.no_grad():
            for imgs, targets in self.val_loader:
                imgs = list(img.to(DEVICE) for img in imgs)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(imgs, targets)
                loss = sum( loss for loss in loss_dict.values())
                total_loss.append(loss)
        avg_loss=sum(total_loss)/len(total_loss)
        self.lr_scheduler.step(avg_loss)
        if current_lr!=get_lr(self.optimizer):
            logger.info("Loading best model weights")
            self.model.load_state_dict(self.best_weights)
        
        if sum(total_loss)<self.best_validation_loss:
            logger.info("Updating best model weights")
            self.best_validation_loss=sum(total_loss)
            self.best_weights=copy.deepcopy(self.model.state_dict())
        
        return avg_loss
        
        

def main():

    ins = Trainer()
    ins.run()

if __name__ == '__main__':
    main()
                                    
        
        
        
        
    


