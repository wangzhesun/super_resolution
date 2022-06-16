# EDSR Super Resolution in PyTorch

This repository contains a clean and clear PyTorch implementation of the EDSR algorithm described in the
CVPR2017 workshop Paper: ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/pdf/1707.02921.pdf) 
for both CPU-only and GPU users. The official implementation of the algorithm can be found
[here](https://github.com/sanghyun-son/EDSR-PyTorch).

## Usage
The EDSR system of this project comprises three stages: data preprocessing, training,
and evaluation. It's totally fine to skip the first two parts and go straight to the evaluation for a
quick start. In that case, refer to the evaluation section directly to get started (pre-trained weight 
files are provided in that section).

### Preprocessing
The dataset used in this project is [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf), which can be downloaded [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar). Other datasets are welcome to use.
You can place the dataset wherever you want.

After getting the dataset, use the `augment.py` in the `preprocess` directory to augment the dataset.
This script will augment the dataset into smaller but more images using random image processing operations
(e.g., flipping, rotation). You may want to change the parameters to where the dataset is located and where
you want the augmented dataset goes.

An example of the `augment.py` usage is shown as follows:

```
util.augment_dir(train_root='./data/DIV2K/DIV2k_train_LR_bicubic/X4', target_root='./data/DIV2K/DIV2k_train_HR', 
                 train_output_path='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug',
                 target_output_path='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                 aug_num=20, hr_crop_size=192, scale=4)
```

For your convenience, the scale-4 augmented dataset 
using `DIV2K_train_HR` and `DIV2K_train_LR_bicubic/X4` is provided [here](https://drive.google.com/drive/folders/1gD_y0ZXxPIdJbnLRDgOaf7KLNbJ6hKNA?usp=sharing).


### Training
To train your model, use the script `train.py`. Change the parameters to where you store the dataset.
You can also customize the training process by changing the epoch, learning rate, etc. Depending on your
situation, you may run in CPU (set `cuda` to be `False`) or in GPU (set `cuda` to be `True`) setting. 
For each epoch, the checkpoint will be saved in the directory named `checkpoints`, which will be created
during the first epoch if it does not exist.

An example of the `train.py` usage is shown as follows:

```
train_data = Dataset(train_root='./data/DIV2K_aug/DIV2K_train_LR_bicubic_X4_aug', target_root='./data/DIV2K_aug/DIV2K_train_HR_X4_aug',
                     transform=ToTensor())
                     
train_model(scale=4, train_dataset=train_data, epoch=100, lr=1e-4, batch_size=16,
            checkpoint_save_path='checkpoints', checkpoint=False, checkpoint_load_path=None,
            cuda=False)
```

### Evaluation
You can evaluate the model using the `evaluate.py` script. To use your own model, change the
parameters `image_path` and `weight_path` accordingly, and set `pre_train` to `False`.

For a quick start,
you can also use our pre-trained models with scales 2, 3, and 4, respectively. The pre-trained models can be downloaded
from [here](https://drive.google.com/drive/folders/1ok75nwikHz_ODhYiofIwFJSwid9j8uxe?usp=sharing). To use
the pre-trained model, just place three weight files in the `weights` directory and set the parameter `pre_train` to
`True`, then you are good to go.

An example of the `evaluate.py` usage is shown as follows:

```
enhance(scale=4, image_path='./demos/bird.png', pre_train=True, weight_path=None, display=True,
        save=True, output_path='results', cuda=False)
```
