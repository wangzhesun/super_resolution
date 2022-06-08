import torch
import os


def adjust_lr(lr, epoch, step):
    return lr * (0.5 ** (epoch // step))


def save_checkpoint(model, epoch, path):
    model_out_path = path + "/model_epoch_{}.pt".format(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
