import torch
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import glob
import os


def tensor_to_numpy(img):
    """
    convert tensor image to numpy image (0-1 scale to 0-255 scale)

    :param img: the input tensor image
    :return: the output numpy image
    """
    return 255 * img.permute(1, 2, 0).numpy()


def show_tensor_img(img, img_name):
    """
    display the input tensor image

    :param img: the input image
    :param img_name: the image name
    """
    # convert the tensor image to numpy image
    numpy_img = tensor_to_numpy(img)
    numpy_img[numpy_img < 0] = 0
    numpy_img[numpy_img > 255.] = 255.
    numpy_img = numpy_img.astype(np.uint8)
    # convert the image from BGR representation to RGB representation
    corrected_img = cv.cvtColor(numpy_img, cv.COLOR_BGR2RGB)
    plt.imshow(corrected_img)
    plt.title(img_name)
    plt.show()


def save_tensor_img(img, img_name, output_path):
    """
    save the tensor image to the output path

    :param img: the input image
    :param img_name: the image name
    :param output_path: the output path
    """
    numpy_img = tensor_to_numpy(img)  # convert the tensor image to numpy image
    output_p = '{}/{}.jpg'.format(output_path, img_name)
    cv.imwrite(output_p, numpy_img)


def crop(lr_img, hr_img, data_type='tensor', hr_crop_size=192, scale=4):
    """
    crop the same region in low-resolution and high-resolution images

    :param lr_img: the low resolution image
    :param hr_img: the high resolution image
    :param data_type: flag indicating whether the input images are tensor images or numpy images
    :param hr_crop_size: the cropped size of the high resolution image, default is 192
    :param scale: scale of the high-resolution image compared to the low-resolution image
    :return: the cropped low-resolution and high-resolution images
    """
    lr_crop_size = hr_crop_size // scale  # compute the crop size of the low-resolution image
    if data_type == 'tensor':
        lr_img_shape = lr_img.shape[1:3]
    elif data_type == 'array':
        lr_img_shape = lr_img.shape[0:2]
    else:
        raise NotImplementedError

    # determine the width and height of cropped low-resolution and high-resolution images
    lr_width = torch.randint(low=0, high=lr_img_shape[1] - lr_crop_size + 1, size=(1, 1)).item()
    lr_height = torch.randint(low=0, high=lr_img_shape[0] - lr_crop_size + 1, size=(1, 1)).item()
    hr_width = lr_width * scale
    hr_height = lr_height * scale

    # extract the corresponding cropped parts
    if data_type == 'tensor':
        lr_img_cropped = lr_img[:, lr_height:lr_height + lr_crop_size,
                         lr_width:lr_width + lr_crop_size]
        hr_img_cropped = hr_img[:, hr_height:hr_height + hr_crop_size,
                         hr_width:hr_width + hr_crop_size]
    elif data_type == 'array':
        lr_img_cropped = lr_img[lr_height:lr_height + lr_crop_size,
                         lr_width:lr_width + lr_crop_size, :]
        hr_img_cropped = hr_img[hr_height:hr_height + hr_crop_size,
                         hr_width:hr_width + hr_crop_size, :]
    else:
        raise NotImplementedError

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img, data_type='tensor'):
    """
    randomly flip the images left-right

    :param lr_img: the low-resolution image
    :param hr_img: the high-resolution image
    :param data_type: flag indicating whether input images are tensor images or numpy images
    :return: original images or flipped images
    """
    random = torch.rand(1).item()  # generate a random number between 0 and 1

    if random < 0.5:  # only flip if the generated random is greater or equal to 0.5
        return lr_img, hr_img
    else:
        if data_type == 'tensor':
            return torch.flip(lr_img, (2,)), torch.flip(hr_img, (2,))
        elif data_type == 'array':
            return cv.flip(lr_img, 1), cv.flip(hr_img, 1)
        else:
            raise NotImplementedError


def random_rotate(lr_img, hr_img, data_type='tensor'):
    """
    randomly rotate the images by 0, 90, 180, or 270 degrees clockwise

    :param lr_img: the low-resolution image
    :param hr_img: the high-resolution image
    :param data_type: flag indicating whether input images are tensor images or numpy images
    :return: original images or rotated images
    """
    # generate a random integer among 0, 1, 2, and 3
    random = torch.randint(low=0, high=4, size=(1, 1)).item()
    # rotate according to the generated random number
    if data_type == 'tensor':
        return torch.rot90(lr_img, random, (1, 2)), torch.rot90(hr_img, random, (1, 2))
    elif data_type == 'array':
        out_lr = lr_img
        out_hr = hr_img
        for _ in range(random):
            out_lr = cv.rotate(out_lr, cv.ROTATE_90_CLOCKWISE)
            out_hr = cv.rotate(out_hr, cv.ROTATE_90_CLOCKWISE)
        return out_lr, out_hr
    else:
        raise NotImplementedError


def augment_image(train_img_file, target_img_file, train_output_path, target_output_path,
                  aug_num, hr_crop_size, scale):
    """
    augment a single image for some times

    :param train_img_file: the path to the training image
    :param target_img_file: the path to the target image
    :param train_output_path: the output path of the augmented training image
    :param target_output_path: the output path of the augmented target image
    :param aug_num: the repeat number of the augmentation process
    :param hr_crop_size: the cropped size of the high resolution image
    :param scale: scale of the high-resolution image compared to the low-resolution image
    """
    train_img = cv.imread(train_img_file)
    target_img = cv.imread(target_img_file)
    # repeat the augmentation process for aug_num times
    for i in range(aug_num):
        count = 0  # record the number of times when no operation applied
        c_train, c_target = crop(train_img, target_img, data_type='array',
                                 hr_crop_size=hr_crop_size, scale=scale)

        random = torch.torch.rand(1).item()
        if random < 0.5:
            f_c_train, f_c_target = random_flip(c_train, c_target, data_type='array')
        else:
            f_c_train, f_c_target = c_train, c_target
            count += 1

        random = torch.torch.rand(1).item()
        if random < 0.5:
            r_f_c_train, r_f_c_target = random_rotate(f_c_train, f_c_target, data_type='array')
        else:
            r_f_c_train, r_f_c_target = f_c_train, f_c_target
            count += 1

        # output the augmented image only if some operation has been applied
        if count < 2:
            train_out = '{}/{}_aug_{}.png'.format(train_output_path,
                                                  train_img_file.split('/')[-1].split('\\')[-1]
                                                  .split('.')[0], i + 1)
            target_out = '{}/{}_aug_{}.png'.format(target_output_path,
                                                   target_img_file.split('/')[-1].split('\\')[-1]
                                                   .split('.')[0], i + 1)
            cv.imwrite(train_out, r_f_c_train)
            cv.imwrite(target_out, r_f_c_target)


def augment_dir(train_root, target_root, train_output_path, target_output_path, aug_num=5,
                hr_crop_size=192, scale=4):
    """
    augment the whole directory for some times

    :param train_root: the path to the training dataset directory
    :param target_root: the path to the training dataset directory
    :param train_output_path: the output path of the augmented training dataset
    :param target_output_path: the output path of the augmented target dataset
    :param aug_num: the repeat number of the augmentation process
    :param hr_crop_size: the cropped size of the high resolution image
    :param scale: scale of the high-resolution image compared to the low-resolution image
    """
    train_files = np.array(glob.glob(train_root + '/*'))
    target_files = np.array(glob.glob(target_root + '/*'))

    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    if not os.path.exists(target_output_path):
        os.makedirs(target_output_path)

    for i in range(len(train_files)):
        print('augmenting image {} ...'.format(i + 1))
        augment_image(train_files[i], target_files[i], train_output_path, target_output_path,
                      aug_num=aug_num, hr_crop_size=hr_crop_size, scale=scale)
