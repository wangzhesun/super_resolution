import torch.utils.data as data
import cv2 as cv
import glob
import numpy as np
import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, train_path, target_path, transform=None):
        super(Dataset, self).__init__()
        self.transform_ = transform
        self.train_ = np.array(glob.glob(train_path + '/*'))
        self.target_ = np.array(glob.glob(target_path + '/*'))
        self.train_data = {}
        self.target_data = {}
        assert len(self.train_) == len(self.target_)

    def __getitem__(self, index):
        if index in self.train_data:
            return self.train_data[index], self.target_data[index]
        train_index_img = cv.imread(self.train_[index])
        target_index_img = cv.imread(self.target_[index])
        if self.transform_:
            train_index_img = self.transform_(train_index_img)
            target_index_img = self.transform_(target_index_img)
        self.train_data[index] = train_index_img
        self.target_data[index] = target_index_img
        return train_index_img, target_index_img

    def __len__(self):
        return self.train_.shape[0]

    def display(self, label='all', index=0):
        if label == 'train' or label == 'all':
            train_img = self[index][0].permute(1, 2, 0)
            cv.imshow('train image', train_img.numpy())
            if label == 'all':
                target_img = self[index][1].permute(1, 2, 0)
                cv.imshow('target image', target_img.numpy())
        elif label == 'target' or label == 'all':
            if label == 'all':
                train_img = self[index][0].permute(1, 2, 0)
                cv.imshow('train image', train_img.numpy())
            target_img = self[index][1].permute(1, 2, 0)
            cv.imshow('target image', target_img.numpy())
        else:
            raise NotImplementedError

        cv.waitKey(0)


if __name__ == '__main__':
    train_path = './data/DIV2K_train_LR_bicubic/X4'
    target_path = './data/DIV2K_train_HR'

    a = Dataset(train_path, target_path, transform=transforms.ToTensor())
    a.display(label='train', index=2)
    a.display(label='train', index=2)
    a.display(label='train', index=2)
    a.display(label='train', index=3)
    a.display(label='train', index=3)
