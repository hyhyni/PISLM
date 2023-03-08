# Mining High-quality Pseudo Instance Soft Labels for Weakly Supervised Object Detection in Remote Sensing Images
By Xiaoliang Qian, Yu Huo, Gong Cheng, Chenyang Gao, Xiwen Yao, Wei Wang

## Requirements
* python == 3.7 <br>
* Cuda == 10.1 <br>
* Pytorch == 1.7.1 <br>
* torchvision == 0.8.2 <br>
* apex <br>
* ninja <br>
* cython <br>
* opencv <br>
* tensorboardX <br>
* pycocotools <br>
* GPU: NVIDIA TITAN RTX

## Installation

1. Clone the pqs-psla repository
```bash
git clone https://github.com/
cd pqs-psla
```
2. Environment installation and compilation
```bash
conda create --name pqs-psla python=3.7
conda activate pqs-psla
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
python setup.py build develop
```
3. Download the VOCdevkit and rename it as VOCdevkit2007
```bash
cd RINet/data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
4. Download the training, validation, test data from [NWPU](https://onedrive.live.com/?authkey=%21ADaUNysmiFRH4eE&cid=5C5E061130630A68&id=5C5E061130630A68%21115&parId=5C5E061130630A68%21113&action=locate) and [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
5. Extract all of datasets into one directory named VOCdevkit2007
6. Download pretrained ImageNet weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it in the data/imagenet_weights/



