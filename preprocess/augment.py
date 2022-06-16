from utils import image_util as util

if __name__ == '__main__':
    # make sure you change the train_root to the path of the raw training data directory,
    # the target_root to the path of the raw target directory, the train_output_path to the desired
    # output path of augmented training data, and the target_output_path to the desired output path
    # of augmented target data
    util.augment_dir(train_root='raw_train_dataset_path',
                     target_root='raw_target_dataset_path',
                     train_output_path='augmented_train_dataset_output_path',
                     target_output_path='augmented_target_dataset_output_path',
                     aug_num=20, hr_crop_size=192, scale=4)
