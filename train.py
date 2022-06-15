from utils.dataset import Dataset
from torchvision.transforms import ToTensor
from utils.model_util import train_model


if __name__ == '__main__':
    # load dataset
    print('loading dataset ...')
    # make sure you change the train_root to the path of the training data directory,
    # and change the target_root to the path of the target directory
    train_data = Dataset(train_root='train_data_path', target_root='target_data_path',
                         transform=ToTensor())

    # train the model
    train_model(scale=4, train_dataset=train_data, epoch=100, lr=1e-4, batch_size=16,
                checkpoint_save_path='checkpoints', checkpoint=False, checkpoint_load_path=None,
                cuda=False)
