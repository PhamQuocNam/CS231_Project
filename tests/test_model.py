from models import FasterRCNNModel
import torch

def test_model_forward():
    model = FasterRCNNModel()
    x = torch.randn(3,3,600,600)  
    assert model(x) , "forward error!"
