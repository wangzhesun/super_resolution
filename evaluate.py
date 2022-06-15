from utils.model_util import enhance


if __name__ == '__main__':
    enhance(scale=2, image_path='input_image_path/input_image_name', pre_train=True, display=True,
            save=True, output_path='results', cuda=False)
