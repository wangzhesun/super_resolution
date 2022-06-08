import image_util as util

if __name__ == '__main__':
    util.augment_dir(train_root='./data/DIV2K/DIV2K_train_LR_bicubic/X4',
                     target_root='./data/DIV2K/DIV2K_train_HR',
                     train_output_path='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                     target_output_path='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                     aug_num=10, hr_crop_size=96, scale=4)
