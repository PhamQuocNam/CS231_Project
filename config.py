import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "data/"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

# Model Parameters
MODEL_NAME='FasterRCNN'
INPUT_CHANNELS = 3
NUM_CLASSES = 10
CHECKPOINT_FILE = "checkpoints/best_model.pth/FasterRCNN_model_epoch1.pth"

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
# Logging
LOG_INTERVAL = 10

# Inference
SOURCE_PATH= "data_source"
RESULT_PATH = "results"