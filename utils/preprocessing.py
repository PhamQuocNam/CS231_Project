import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from config import DATA_DIR
transform = {
    'train': v2.Compose([
        v2.ToImage(),                                # Convert PIL to tensor
        v2.Resize((600, 600), antialias=True),       # Resize with antialiasing
        v2.RandomHorizontalFlip(p=0.5),              # More common to use p=0.5
        v2.ColorJitter(brightness=0.3, contrast=0.2,saturation =0.3),
        v2.RandomVerticalFlip(p=0.5),
        # v2.RandomRotation(degrees=(-90, 90)),        # Rotates both image and boxes
        v2.ToDtype(torch.float32, scale=True),       # Convert to float and scale to [0,1]
    ]),
    'test': v2.Compose([
        v2.ToImage(),
        # v2.RandomCrop((224,224)),
        v2.Resize((600, 600), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    'val': v2.Compose([
        v2.ToImage(),
        # v2.RandomCrop((224,224)),
        v2.Resize((600, 600), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]),
}


def collate_fn(batch):
    return tuple(zip(*batch))


def check(dataset, n_samples):
    for i in range(n_samples):
        img, target= dataset[i]
        boxes = target['boxes']
        boxes=boxes.detach().numpy().astype(np.int32)
        sample=img.permute(1,2,0).numpy().copy()
        for box in boxes:
            sample=cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 3)
        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.imshow(sample)
        plt.savefig(f"image_test/image{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn)

def data_splitting(X,y):
    X_train,X_val, y_train,y_val = train_test_split(X,y, test_size=0.4, shuffle=True)
    X_val,X_test, y_val, y_test = train_test_split(X_val,y_val, test_size=0.5, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

        
def extract_files(file_path):
    img_files = []
    label_files = []
    for file in os.listdir(os.path.join(DATA_DIR,file_path)):
        if file.split('.')[-1] == 'jpeg' or file.split('.')[-1] == 'jpg':
            img_files.append(os.path.join(DATA_DIR + '/' + file_path,file))
        elif file != 'count.txt':
            label_files.append(os.path.join(DATA_DIR + '/'+ file_path,file))
            
    assert len(img_files) == len(label_files), "Mismatch between images and labels"
    return sorted(img_files), sorted(label_files)