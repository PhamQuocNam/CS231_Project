import torch
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(model_name ,model_state, optimizer, epoch, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_dir / f"{model_name}_model_epoch{epoch}.pth")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]


def get_lr(optimizer):
    for params in optimizer.param_groups:
        return params["lr"]
    
    
def training_visualizing(history):
    assert history['training']== history['valid'], "Training and validation loss lengths must match."
    
    epochs = range(1,len(history['training']))
    # Plotting
    training_loss=np.array(history['training'])
    valid_loss=[sample.cpu().numpy() for sample in history['valid']]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label='Training Loss', marker='o')
    plt.plot(epochs, valid_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_visualizing.png")
    plt.close()