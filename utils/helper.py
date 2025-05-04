import torch

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
