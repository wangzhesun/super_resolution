import cv2 as cv
import numpy as np
import math
from utils import image_util as util


def get_psnr(img1, img2, shave_border=0, tensor=False):
    """
    compute the PSNR score between two images

    :param img1: the first image
    :param img2: the second image
    :param shave_border: the amount of border to be shaved
    :param tensor: flag indicating whether input images are tensor images or numpy images
    :return: the PSNR score
    """
    if tensor:
        img1 = util.tensor_to_numpy(img1)
    else:
        img1 = img1.astype(int)

    if tensor:
        img2 = util.tensor_to_numpy(img2)
    else:
        img2 = img2.astype(int)

    height, width = img1.shape[0:2]

    # crop both images to the region we should focus on
    img1 = img1[shave_border:height - shave_border,
           shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border,
           shave_border:width - shave_border]

    im_dff = img1 - img2
    rmse = math.sqrt(np.mean(im_dff ** 2))

    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def ssim(img1, img2):
    """
    helper function of get_ssim to compute the SSIM score of a single channel

    :param img1: the first image
    :param img2: the second image
    :return: SSIM score of a single channel
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def get_ssim(img1, img2, tensor=False):
    """
    compute the SSIM score between two images

    :param img1: the first image
    :param img2: the second image
    :param tensor: flag indicating whether input images are tensor images or numpy images
    :return: the SSIM score
    """
    if tensor:
        img1 = util.tensor_to_numpy(img1)
    else:
        img1 = img1.astype(int)

    if tensor:
        img2 = util.tensor_to_numpy(img2)
    else:
        img2 = img2.astype(int)

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:  # in case both images are single-channel images
        return ssim(img1, img2)
    elif img1.ndim == 3:  # in case both images are multi-channel images
        if img1.shape[2] == 3:  # for three-channel image
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:  # for single-channel image
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
