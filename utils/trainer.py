import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utils import model_util as util


class Trainer:
    def __init__(self, dataset, model, cuda):
        self.dataset_ = dataset
        self.model_ = model
        self.step = 10
        self.checkpoint_path_ = None
        self.cuda_ = cuda

    def set_checkpoint_saving_path(self, path):
        self.checkpoint_path_ = path

    def train(self, epoch, batch_size=16, lr=1e-4, num_worker=3, weight_decay=1e-4,
              checkpoint_load_path=None, checkpoint=False):
        train_loader = DataLoader(self.dataset_, num_workers=num_worker, batch_size=batch_size,
                                  shuffle=True)

        start_epoch = 0
        if checkpoint:
            if os.path.isfile(checkpoint_load_path):
                print('loading checkpoint \'{}\' ...'.format(checkpoint_load_path))
                checkpoint = torch.load(checkpoint_load_path)
                state = checkpoint['model']
                lr = checkpoint['lr']
                start_epoch = checkpoint['epoch']
                self.model_.load_state_dict(state)
                print('loading checkpoint successfully')
            else:
                print('=> no checkpoint found at \'{}\''.format(checkpoint_load_path))

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model_.parameters()), lr=lr,
                               weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-08)

        criterion = nn.L1Loss(reduction='sum')
        if self.cuda_:
            criterion = criterion.cuda()

        print('training ...')
        for epoch_i in range(start_epoch, start_epoch+epoch+1):
            lr = util.adjust_lr(lr, epoch - 1, self.step)
            check_count = 1

            for i, (input_img, target_img) in enumerate(train_loader):
                if self.cuda_:
                    input_img = input_img.cuda()
                    target_img = target_img.cuda()
                predict_img = self.model_(input_img)
                loss = criterion(predict_img, target_img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('===> Epoch[{}/{}]({}/{}): Loss: {}'.format(epoch_i + 1, start_epoch+epoch+1,
                                                                  i+1, len(train_loader),
                                                                  round(loss.item(), 1)))

                if check_count % 1 == 0:
                    util.save_checkpoint(model=self.model_, lr=lr, epoch=epoch_i + 1,
                                         mark=check_count // 1, path=self.checkpoint_path_)
                check_count += 1

                if i == 1:
                    break

        util.save_checkpoint(model=self.model_, lr=lr, epoch=epoch, path=self.checkpoint_path_,
                             final=True)
