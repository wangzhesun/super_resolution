from utils import dataset
import torchvision.transforms as transforms
from utils.trainer import Trainer
from utils import model_util as util


def train_model(model, train_dataset, epoch, checkpoint_save_path, checkpoint=False,
                checkpoint_load_path=None, cuda=False):
    trainer = Trainer(train_dataset, model, cuda=cuda)
    trainer.set_checkpoint_saving_path(checkpoint_save_path)
    trainer.train(epoch=epoch, checkpoint_load_path=checkpoint_load_path,
                  checkpoint=checkpoint)


if __name__ == '__main__':
    # load dataset
    print('loading dataset ...')
    train_data = dataset.Dataset(train_root='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                                 target_root='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                                 transform=transforms.ToTensor())

    # create the model
    print('initializing model ...')
    sr_model = util.create_model(scale=4)

    # train the model
    train_model(model=sr_model, train_dataset=train_data, epoch=100,
                checkpoint_save_path='./checkpoints', checkpoint=False,
                checkpoint_load_path=None, cuda=False)
