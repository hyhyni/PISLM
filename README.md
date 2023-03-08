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
3. Download the DIOR dataset and put them into the ./datasets directory. For example:
```bash
./datasets/voc/VOC2007/JPEGImages
./datasets/voc/VOC2007/Annotations
./datasets/voc/VOC2007/ImageSets
```
4. Download selective search proposals from [DIOR](https://onedrive.live.com/), and put them into the ./proposal directory.
5. Download pretrained weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it into the ./pretrained_weights directory.
## Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py --config-file "configs/voc/V_16_voc07.yaml" \
--use-tensorboard OUTPUT_DIR ./output/dir
```
## Test
```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/test_net.py --config-file "configs/voc/V_16_voc07.yaml" TEST.IMS_PER_BATCH 2 \
OUTPUT_DIR ./output/test_output MODEL.WEIGHT ./output/dir/model
```
## Acknowledgement
We borrowed the main code from [wetectron](https://github.com/NVlabs/wetectron).
