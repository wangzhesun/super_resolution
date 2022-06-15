from utils.model_util import enhance


if __name__ == '__main__':
    # make sure you change the image_path to the path of the input image
    enhance(scale=2, image_path='input_image_path/input_image_name', pre_train=True, display=True,
            save=True, output_path='results', cuda=False)
