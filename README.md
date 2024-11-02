# PIM-Net

**PIM-Net: Progressive Inconsistency Mining Network for Image Manipulation Localization**  (_Pattern Recognition 2024_)

## Environment
* Python 3.7.13
* Pytorch 1.11.0
* Tensorboard 2.8.0
* OPCV 4.6.0
* CUDA 11.3 + cudnn 8.2.0

## Train
Train code: <train.py>

## Test
Test code: <test.py>

## Metrics
Five evaluation metrics (F1, AUC, IoU, MCC, FPR) are calculated by running <metric5.py>.

## Test Dataset
Please refer to [PSCC-Net](https://github.com/proteus1991/PSCC-Net) for the test dataset.

## Train Dataset
[TIP 2019](https://github.com/jawadbappy/forgery_localization_HLED) We randomly selected 27,639 splicing images.

## Visualizations

![columbia](https://github.com/ningnbai/PIM-Net/assets/106603827/43fe7b3a-311c-4a8e-a4d2-88c63e0dabec)
![casia](https://github.com/ningnbai/PIM-Net/assets/106603827/bb8be896-ada2-4281-a7be-f9bd0154e6ca)
![nist16](https://github.com/ningnbai/PIM-Net/assets/106603827/a0600114-82ba-4af1-bd47-ba97ef50e2e4)
![cover](https://github.com/ningnbai/PIM-Net/assets/106603827/beaf78a5-77aa-4d13-8880-f8c06d367aaa)
![imd20](https://github.com/ningnbai/PIM-Net/assets/106603827/c97cf472-1e83-4600-ad74-4fe4cfb2b1ac)
