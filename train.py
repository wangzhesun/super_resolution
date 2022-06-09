from utils import dataset
import torchvision.transforms as transforms
from utils import model_util as util


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
    util.train_model(model=sr_model, train_dataset=train_data, epoch=100,
                     checkpoint_save_path='checkpoints', checkpoint=False,
                     checkpoint_load_path=None, cuda=False)
