from utils.model_util import enhance


if __name__ == '__main__':
    enhance(scale=4, image_path='./data/DIV2K/DIV2K_test_LR_unknown/X4/0904x4.png',
            weight_path='./checkpoints/model_epoch_7_save_1.pt', display=True, save=True,
            output_path='./tmp', cuda=False)

