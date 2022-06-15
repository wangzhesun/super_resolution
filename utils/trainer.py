import torch.autograd
from torch import optim, nn
from torch.utils.data import DataLoader

from utils import model_util as util


class Trainer:
    def __init__(self, dataset, model, cuda, lr=1e-4, batch_size=16):
        self.dataset_ = dataset
        self.cuda_ = cuda
        self.model_ = model
        self.step_ = 2e5
        self.lr_ = lr
        self.batch_size_ = batch_size
        self.batch_update_ = 1
        self.checkpoint_path_ = None

    def set_checkpoint_saving_path(self, path):
        """
        set the output path for checkpoints

        :param path: the output path for checkpoints
        """
        self.checkpoint_path_ = path

    def train(self, epoch, num_worker=1, weight_decay=1e-4, checkpoint_load_path=None,
              checkpoint=False):
        """
        train the model

        :param epoch: the epoch number for the training
        :param num_worker: number of workers for the dataloader
        :param weight_decay: weight decay for the optimizer
        :param checkpoint_load_path: the output path for the checkpoints
        :param checkpoint: flag indicating whether a pretrained checkpoint will be used
        """
        train_loader = DataLoader(self.dataset_, num_workers=num_worker,
                                  batch_size=self.batch_size_, shuffle=True)

        start_epoch = 0
        if checkpoint: # load the checkpoint if needed
            self.model_, start_epoch, self.lr_, self.batch_update_, _ = \
                util.load_checkpoint(self.model_, checkpoint_load_path)

        # set up the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model_.parameters()),
                               lr=self.lr_, weight_decay=weight_decay, betas=(0.9, 0.999),
                               eps=1e-08)

        criterion = nn.L1Loss()
        if self.cuda_:
            criterion = criterion.cuda()

        print('training ...')
        for epoch_i in range(start_epoch, start_epoch + epoch):
            self.model_.train()
            for i, (input_img, target_img) in enumerate(train_loader):
                # update the learning rate and optimizer for every "step"-size updates
                if self.batch_update_ % self.step_ == 0:
                    self.lr_ /= 2
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                                  self.model_.parameters()),
                                           lr=self.lr_, weight_decay=weight_decay,
                                           betas=(0.9, 0.999), eps=1e-08)
                # convert the tensor image from 0-1 to 0-255
                input_img *= 255
                target_img *= 255

                if self.cuda_:
                    input_img = input_img.cuda()
                    target_img = target_img.cuda()
                predict_img = self.model_(input_img)
                target_img = torch.autograd.Variable(target_img, requires_grad=False)
                loss = criterion(predict_img, target_img)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.batch_update_ += 1

                print('===> Epoch[{}/{}]({}/{}): Loss: {}'.format(epoch_i + 1, start_epoch + epoch,
                                                                  i + 1, len(train_loader),
                                                                  round(loss.item(), 4)))
            # save the checkpoint for every epoch
            util.save_checkpoint(model=self.model_, epoch=epoch_i + 1, lr=self.lr_,
                                 batch=self.batch_update_, path=self.checkpoint_path_)

        util.save_checkpoint(model=self.model_, epoch=epoch, batch=self.batch_update_, lr=self.lr_,
                             path=self.checkpoint_path_, final=True)
