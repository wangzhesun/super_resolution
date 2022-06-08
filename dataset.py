import torch.utils.data as data
import cv2 as cv
import glob
import numpy as np
import image_util as util


class Dataset(data.Dataset):
    def __init__(self, train_root, target_root='./data/DIV2K/DIV2K_train_HR', transform=None):
        super(Dataset, self).__init__()
        self.transform_ = transform
        self.train_ = np.array(glob.glob(train_root + '/*'))
        self.target_ = np.array(glob.glob(target_root + '/*'))
        assert len(self.train_) == len(self.target_)

    def __getitem__(self, index):
        train_index_img = cv.imread(self.train_[index])
        target_index_img = cv.imread(self.target_[index])
        if self.transform_:
            train_index_img = self.transform_(train_index_img)
            target_index_img = self.transform_(target_index_img)
        return train_index_img, target_index_img

    def __len__(self):
        return self.train_.shape[0]

    def display(self, label='all', index=0):
        if label == 'train' or label == 'all':
            util.show_tensor_img(self[index][0], 'train image')
            if label == 'all':
                util.show_tensor_img(self[index][1], 'target image')
        elif label == 'target' or label == 'all':
            if label == 'all':
                util.show_tensor_img(self[index][0], 'train image')
            util.show_tensor_img(self[index][1], 'target image')
        else:
            raise NotImplementedError

        cv.waitKey(0)
