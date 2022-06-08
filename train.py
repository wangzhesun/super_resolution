from model import edsr
from utils import dataset
import torchvision.transforms as transforms
from utils import trainer

if __name__ == '__main__':
    # load dataset
    print('loading dataset ...')
    train_data = dataset.Dataset(train_root='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                                 target_root='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                                 transform=transforms.ToTensor())

    # initialize model
    print('initializing model ...')
    EDSR = edsr.EDSR(scale=4)

    # train the model
    trainer = trainer.Trainer(train_data, EDSR)
    trainer.set_checkpoint_saving_path('./checkpoints')
    trainer.train(epoch=2, checkpoint_load_path='./checkpoints/model_epoch_1_save_9.pt',
                  checkpoint=True)
