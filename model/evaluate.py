from utils import model_util
from utils import image_util


if __name__ == '__main__':
    print('initializing model ...')
    sr_model = model_util.create_model(scale=4)

    sr_model, _, _ = model_util.load_checkpoint(sr_model, '../checkpoints/model_epoch_7_save_1.pt')

    print('enhancing image ...')
    input_img, output_img = \
        model_util.evaluate_img(sr_model, '../data/DIV2K/DIV2K_test_LR_unknown/X4/0904x4.png')

    print('image displayed')
    image_util.show_tensor_img(input_img.detach(), 'input image')
    image_util.show_tensor_img(output_img.detach(), 'output image')
