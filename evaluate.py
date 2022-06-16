from utils.model_util import enhance


if __name__ == '__main__':
    # make sure you change the image_path to the path of the input image
    enhance(scale=4, image_path='input_image_path', pre_train=True, display=True,
            save=True, output_path='results', cuda=False)
