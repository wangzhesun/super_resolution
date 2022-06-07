import util

if __name__ == '__main__':
    util.augment_dir(train_root='./data/DIV2K/DIV2K_train_LR_bicubic/X4',
                     target_root='./data/DIV2K/DIV2K_train_HR',
                     train_output_path='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                     target_output_path='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                     aug_num=30, hr_crop_size=96, scale=4)
    # util.augment_dir(train_root='./tmp_train',
    #                  target_root='./tmp_target',
    #                  train_output_path='./tmp_train_out',
    #                  target_output_path='./tmp_target_out',
    #                  aug_num=10, hr_crop_size=96, scale=4)
