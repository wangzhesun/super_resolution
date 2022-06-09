from utils import model_util
from utils import image_util
import cv2 as cv
import torchvision.transforms as transforms


def evaluate_img(model, input_path, cuda=False):
    input_image = transforms.ToTensor()(cv.imread(input_path))
    if cuda:
        input_image = input_image.cuda()

    output_image = model(input_image)
    return output_image


if __name__ == '__main__':
    sr_model = model_util.create_model(scale=4)
    sr_model = model_util.load_checkpoint(sr_model, './checkpoints/model_epoch_7_save_1.pt')

    output = evaluate_img(sr_model, './data/DIV2K/DIV2K_test_LR_unknown/X4/0904x4.png')
    image_util.show_tensor_img(output, 'output image')
