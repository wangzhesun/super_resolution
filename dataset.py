import torch.utils.data as data
import cv2 as cv
import glob
import numpy as np
import torchvision.transforms as transforms
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

# if __name__ == '__main__':
    # train_path = './data/DIV2K/DIV2K_train_LR_bicubic/X4'

    # a = Dataset(train_root=train_path, transform=transforms.ToTensor())
    # print(len(a))
    # print(len(a[0]))
    # print(a[0][0].shape)
    # a.display(label='train', index=2)
    # a.display(label='train', index=2)
    # a.display(label='train', index=2)
    # a.display(label='train', index=3)
    # a.display(label='train', index=3)

    # img_train = a[0][0]
    # img_target = a[0][1]
    # cropped_train, cropped_target = util.random_crop(img_train, img_target)
    # util.show_tensor_img(cropped_train, 'cropped train')
    # util.show_tensor_img(cropped_target, 'cropped target')

    # util.save_tensor_img(cropped_train, 'cropped_train', './tmp')
    # util.save_tensor_img(cropped_target, 'cropped_target', './tmp')


    # flip_train, flip_target = util.random_flip(img_train, img_target)
    # util.show_tensor_img(img_train, 'original train')
    # util.show_tensor_img(flip_train, 'flip train')

    # rot_train, rot_target = util.random_rotate(img_train, img_target)
    # util.show_tensor_img(img_train, 'original train')
    # util.show_tensor_img(rot_train, 'rotate train')

