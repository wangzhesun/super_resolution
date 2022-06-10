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
