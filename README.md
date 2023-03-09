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
## Installation
1. Clone the pislm repository
```bash
git clone https://github.com/
cd pislm
```
2. Environment installation and compilation
```bash
conda create --name pislm python=3.7
conda activate pislm
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
python setup.py build develop
```
3. Download the [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC) or other remote sensing datasets and put them into the ./datasets directory. For example:
```bash
./datasets/DIOR/JPEGImages
./datasets/DIOR/Annotations
./datasets/DIOR/ImageSets
```
4. Download selective search proposals from [DIOR](https://drive.google.com/drive/folders/1zWgPJhu2XOpRhUYjFB0Qv2dJ-PE01aLq?usp=sharing) or generate proposals by yourself, and put them into the ./proposal directory.
5. Download pretrained weights from [here](https://drive.google.com/drive/folders/19siwLoC_mcLhLiJ-ACNUbgGCrhGLBY9C?usp=sharing), and put it into the ./pretrained_weights directory.
## Training
```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file "configs/voc/V_16_voc07.yaml" \
--use-tensorboard OUTPUT_DIR ./output/dir
```
## Test
```bash
python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file "configs/voc/V_16_voc07.yaml" TEST.IMS_PER_BATCH 8 \
OUTPUT_DIR ./output/test_output MODEL.WEIGHT ./output/dir/model
```
## Acknowledgement
We borrowed the main code from [wetectron](https://github.com/NVlabs/wetectron).
