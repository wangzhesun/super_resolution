# EDSR Super Resolution in PyTorch

This repository contains a clean and clear PyTorch implementation of EDSR algorithm described in
CVPR2017 workshop Paper: ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/pdf/1707.02921.pdf) 
for both CPU-only and GPU users. The official implementation of the algoirthm can be referred 
[here](https://github.com/sanghyun-son/EDSR-PyTorch).

#Usage
The complete running of the system is composed of three stages: data preprocessing, training,
and evaluation. It's totally fine to skip the first two parts and go straight to the evaluation
as long as you have the weight file the model needs (pre-trained weight files are provided for downloading
in the evaluation section).

#Preprocessing
The dataset used in this project is [DIV2K](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)
, which can be downloaded [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar). Other datasets are welcome to use.
You can place the dataset to wherever you want.

After getting the dataset, use the `augment.py` in the `preprocess` directory to augment the dataset.
This script will augment the dataset into smaller but more images using random image processing operations
(e.g. flipping, rotation). You may want to change the parameters to where the dataset is located and where
you want the augmented dataset goes. For your convenience, scale 4 augmented dataset 
using `DIV2K_train_HR` and `DIV2K_train_LR_bicubic/X4` is provided [here](https://drive.google.com/drive/folders/1gD_y0ZXxPIdJbnLRDgOaf7KLNbJ6hKNA?usp=sharing).


#Training
To train your model, use the script `train.py`. Change the parameters to where you stores the dataset.
You can also customize the training process by changing the epoch, learning rate, etc. Depends on your
own situation, you may choose to run in CPU (set `cuda` False) or in GPU (set `cuda` True) setting. 
For each epoch, the checkpoint will be saved in the directory named `checkpoints`, which will be created
if not exists.

#Evaluation
You can 