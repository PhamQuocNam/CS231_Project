from utils.logger import get_logger
import os
from config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_NAME, DATA_DIR, CHECKPOINT_PATH
from utils import extract_files, Dataset, transform, data_splitting, get_dataloader, get_lr, save_checkpoint
import pytest


def test_preprocess_data():
    data_dir = DATA_DIR
    data_files = os.listdir(data_dir)
    total_image_files, total_label_files= [],[]
    
    for file in data_files:
        try:
            image_files, label_files= extract_files(file)
        except Exception as e:
            pytest.fail(f"extract_files() failed on {file}: {str(e)}")
        
        total_image_files += image_files
        total_label_files += label_files
    
    assert len(total_image_files) > 0, "No image files found"
    assert len(total_label_files) > 0, "No label files found"
    assert len(total_image_files) == len(total_label_files), "Mismatch between images and labels"
    
    train_image_files,val_image_files, train_label_files, val_label_files= data_splitting(total_image_files, total_label_files)
        
    train_dataset = Dataset(train_image_files, train_label_files,transform['train']) 
    
    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    batch = next(iter(train_loader), None)
    assert batch is not None, "Train DataLoader did not yield any batch"
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, "Batch format is invalid"

    images, targets = batch
    assert len(images) > 0, "No images in batch"
    assert isinstance(targets, list), "Targets should be a list of dictionaries"
