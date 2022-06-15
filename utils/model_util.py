from model.edsr import EDSR
import torch
import os
from utils.trainer import Trainer
import torchvision.transforms as transforms
import cv2 as cv
from utils import image_util
from utils import evaluation_util
import numpy as np


def create_model(scale):
    return EDSR(scale=scale)


def save_checkpoint(model, lr=-1, epoch=-1, batch=-1, path=None, final=False,
                    cuda=False):
    if final:
        model_out_path = path + '/final_model_weights.pt'
        if not os.path.exists(path):
            os.makedirs(path)

        state = {'epoch': epoch, 'model': model.state_dict(), 'lr': lr, 'batch': batch,
                 'cuda': cuda}
        torch.save(state, model_out_path)

        print('Final checkpoint saved to {}'.format(model_out_path))
    else:
        model_out_path = path + '/model_epoch_{}.pt'.format(epoch)
        if not os.path.exists(path):
            os.makedirs(path)

        state = {'epoch': epoch, 'model': model.state_dict(), 'lr': lr, 'batch': batch,
                 'cuda': cuda}
        torch.save(state, model_out_path)

        print('Checkpoint saved to {}'.format(model_out_path))


def load_checkpoint(model, checkpoint_load_path):
    if os.path.isfile(checkpoint_load_path):
        print('loading checkpoint \'{}\' ...'.format(checkpoint_load_path))
        checkpoint = torch.load(checkpoint_load_path, map_location=lambda storage, loc: storage)
        state = checkpoint['model']
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        batch_update = checkpoint['batch']
        use_cuda = checkpoint['cuda']
        model.load_state_dict(state)
        print('load checkpoint successfully')

        return model, start_epoch, lr, batch_update, use_cuda
    else:
        print('=> no checkpoint found at \'{}\''.format(checkpoint_load_path))

        return None, None, None


def train_model(scale, train_dataset, epoch, lr, batch_size, checkpoint_save_path, checkpoint=False,
                checkpoint_load_path=None, cuda=False):
    print('initializing model ...')
    sr_model = create_model(scale=scale)
    if cuda:
        sr_model.cuda()
    trainer = Trainer(train_dataset, sr_model, cuda=cuda, lr=lr, batch_size=batch_size)
    trainer.set_checkpoint_saving_path(checkpoint_save_path)
    trainer.train(epoch=epoch, checkpoint_load_path=checkpoint_load_path,
                  checkpoint=checkpoint)


def evaluate_model(model, input_path, cuda=False):
    input_image = 255 * transforms.ToTensor()(cv.imread(input_path))
    if cuda:
        input_image = input_image.cuda()

    output_image = model(input_image)
    return input_image.detach(), output_image.detach()


def enhance(scale, image_path, pre_train=False, weight_path=None, display=False, save=False,
            output_path=None, cuda=False):
    print('initializing model ...')
    sr_model = create_model(scale=scale)

    if pre_train:
        if scale == 2:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx2.pt')
        elif scale == 3:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx3.pt')
        elif scale == 4:
            sr_model, _, _, _, _ = load_checkpoint(sr_model, './weights/EDSRx4.pt')
        else:
            raise NotImplementedError
    else:
        sr_model, _, _, _, _ = load_checkpoint(sr_model, weight_path)

    if cuda:
        sr_model.cuda()

    sr_model.eval()

    print('enhancing image ...')
    lr_img, sr_img = \
        evaluate_model(sr_model, image_path, cuda=cuda)

    lr_img /= 255
    sr_img /= 255

    print('getting evaluation scores ...')
    resize_lr_img = image_util.tensor_to_numpy(lr_img).astype(np.uint8)
    resize_lr_img = cv.resize(resize_lr_img, dsize=(sr_img.shape[2], sr_img.shape[1]),
                              interpolation=cv.INTER_CUBIC)
    resize_lr_img = transforms.ToTensor()(resize_lr_img)

    psnr_score = evaluation_util.get_psnr(resize_lr_img, sr_img, tensor=True)
    ssim_score = evaluation_util.get_ssim(resize_lr_img, sr_img, tensor=True)

    if display:
        image_util.show_tensor_img(lr_img, 'low resolution image')
        image_util.show_tensor_img(sr_img, 'super resolution image')

    if save:
        print('saving image ...')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_util.save_tensor_img(sr_img,
                                   '{}_sr_x{}'.format(image_path.split('/')[-1].split('.')[0],
                                                      scale),
                                   output_path)

    print('PSNR score is: {}'.format(round(psnr_score, 2)))
    print('SSIM score is: {}'.format(round(ssim_score, 2)))
