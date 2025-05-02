import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import copy
import torchvision


class FasterRCNNModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        model =  torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        model.backbone.requires_grad_(False)
        self.model = model
        
    def forward(self,X, y=None):
        if self.training and y is not None:
            # Training mode: Expect X and y (targets) for computing losses
            losses = self.model(X, y)
            return losses
        else:
            # Inference mode: Return detection outputs
            outputs = self.model(X)
            return outputs
    
    
    