from model.edsr import EDSR
import torch
import os


def create_model(scale):
    return EDSR(scale=scale)


def adjust_lr(lr, epoch, step):
    return lr * (0.5 ** (epoch // step))


def save_checkpoint(model, lr=1e-4, epoch=-1, mark=-1, path=None, final=False):
    if final:
        model_out_path = path + '/final_model_weights.pt'
        if not os.path.exists(path):
            os.makedirs(path)

        state = {'epoch': epoch, 'model': model.state_dict(), 'lr': lr}
        torch.save(state, model_out_path)
        torch.save(model.state_dict(), model_out_path)

        print('Final checkpoint saved to {}'.format(model_out_path))
    else:
        model_out_path = path + '/model_epoch_{}_save_{}.pt'.format(epoch, mark)
        if not os.path.exists(path):
            os.makedirs(path)

        state = {'epoch': epoch, 'model': model.state_dict(), 'lr': lr}
        torch.save(state, model_out_path)

        print('Checkpoint saved to {}'.format(model_out_path))
