from utils.dataset import Dataset
from torchvision.transforms import ToTensor
from utils.model_util import train_model


if __name__ == '__main__':
    # load dataset
    print('loading dataset ...')
    train_data = Dataset(train_root='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                         target_root='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                         transform=ToTensor())

    # train the model
    train_model(scale=4, train_dataset=train_data, epoch=100,
                checkpoint_save_path='checkpoints', checkpoint=False, checkpoint_load_path=None,
                cuda=False)
