import cv2 as cv
import numpy as np
import math
import image_util as util


def get_psnr(prediction, target, pred_path=False, tar_path=False, shave_border=0, tensor=False):
    if pred_path:
        prediction_img = cv.imread(prediction)
        prediction_img = prediction_img.astype(int)
    else:
        if tensor:
            prediction_img = util.tensor_to_numpy(prediction)
        else:
            prediction_img = prediction.astype(int)

    if tar_path:
        target_img = cv.imread(target)
        target_img = target_img.astype(int)
    else:
        if tensor:
            target_img = util.tensor_to_numpy(target)
        else:
            target_img = target.astype(int)

    height, width = prediction_img.shape[0:2]

    prediction_img = prediction_img[shave_border:height - shave_border,
                     shave_border:width - shave_border]
    target_img = target_img[shave_border:height - shave_border,
                 shave_border:width - shave_border]

    im_dff = prediction_img - target_img
    rmse = math.sqrt(np.mean(im_dff ** 2))

    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def ssim(img1, img2):
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


def get_ssim(img1, img2, pred_path=False, tar_path=False, tensor=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if pred_path:
        img1 = cv.imread(img1)
        img1 = img1.astype(int)
    else:
        if tensor:
            img1 = util.tensor_to_numpy(img1)
        else:
            img1 = img1.astype(int)

    if tar_path:
        img2 = cv.imread(img2)
        img2 = img2.astype(int)
    else:
        if tensor:
            img2 = util.tensor_to_numpy(img2)
        else:
            img2 = img2.astype(int)

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
