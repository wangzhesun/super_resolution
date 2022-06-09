from model.edsr import EDSR
import torch
import os
from utils.trainer import Trainer
import torchvision.transforms as transforms
import cv2 as cv


def create_model(scale):
    return EDSR(scale=scale)


def adjust_lr(lr, epoch, step):
    return lr * (0.1 ** (epoch // step))


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


def load_checkpoint(model, checkpoint_load_path):
    if os.path.isfile(checkpoint_load_path):
        print('loading checkpoint \'{}\' ...'.format(checkpoint_load_path))
        checkpoint = torch.load(checkpoint_load_path)
        state = checkpoint['model']
        lr = checkpoint['lr']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(state)
        print('load checkpoint successfully')

        return model, start_epoch, lr
    else:
        print('=> no checkpoint found at \'{}\''.format(checkpoint_load_path))

        return None, None, None


def train_model(model, train_dataset, epoch, checkpoint_save_path, checkpoint=False,
                checkpoint_load_path=None, cuda=False):
    trainer = Trainer(train_dataset, model, cuda=cuda)
    trainer.set_checkpoint_saving_path(checkpoint_save_path)
    trainer.train(epoch=epoch, checkpoint_load_path=checkpoint_load_path,
                  checkpoint=checkpoint)


def evaluate_img(model, input_path, cuda=False):
    input_image = transforms.ToTensor()(cv.imread(input_path))
    if cuda:
        input_image = input_image.cuda()

    output_image = model(input_image)
    return input_image, output_image
