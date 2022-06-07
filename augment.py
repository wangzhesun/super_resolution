import util

if __name__ == '__main__':
    util.augment_dir(train_root='./tmp_train', target_root='./tmp_target',
                     train_output_path='./tmp_train_out', target_output_path='./tmp_target_out',
                     aug_num=6, hr_crop_size=96, scale=4)
