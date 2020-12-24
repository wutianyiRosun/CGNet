# [CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/pdf/1811.08201.pdf)
## Introduction
The demand of applying semantic segmentation model on mobile devices has been increasing rapidly. Current
state-of-the-art networks have enormous amount of parameters, hence unsuitable for mobile devices, while other small
memory footprint models follow the spirit of classification network and ignore the inherent characteristic of semantic
segmentation. To tackle this problem, we propose a novel Context Guided Network (CGNet), which is a light-weight
and efficient network for semantic segmentation. We first propose the Context Guided (CG) block, which learns the
joint feature of both local feature and surrounding context, and further improves the joint feature with the global context.
Based on the CG block, we develop CGNet which captures contextual information in all stages of the network and
is specially tailored for increasing segmentation accuracy. CGNet is also elaborately designed to reduce the number
of parameters and save memory footprint. Under an equivalent number of parameters, the proposed CGNet significantly outperforms existing segmentation networks. Extensive experiments on Cityscapes and CamVid datasets verify the effectiveness of the proposed approach. Specifically, without any post-processing and multi-scale testing, the proposed CGNet achieves 64.8% mean IoU on Cityscapes with less than 0.5 M parameters.


## Installation
1. Install PyTorch
  - Env: PyTorch\_0.4; cuda\_9.2; cudnn\_7.5; python\_3.6
2. Clone the repository
   ```shell
   git clone https://github.com/wutianyiRosun/CGNet.git 
   cd CGNet
   ```
3. Dataset

  - Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py). It should have this basic structure.
  ```
  ├── cityscapes_test_list.txt
  ├── cityscapes_train_list.txt
  ├── cityscapes_trainval_list.txt
  ├── cityscapes_val_list.txt
  ├── cityscapes_val.txt
  ├── gtCoarse
  │   ├── train
  │   ├── train_extra
  │   └── val
  ├── gtFine
  │   ├── test
  │   ├── train
  │   └── val
  ├── leftImg8bit
  │   ├── test
  │   ├── train
  │   └── val
  ├── license.txt
```
  - Download the [Camvid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) dataset. It should have this basic structure.
  ```
  ├── camvid_test_list.txt
  ├── camvid_train_list.txt
  ├── camvid_trainval_list.txt
  ├── camvid_val_list.txt
  ├── test
  ├── testannot
  ├── train
  ├── trainannot
  ├── val
  └── valannot

  ```
## Train your own model
  
###  For Cityscapes
  1. training on train set
  ```
  python cityscapes_train.py --gpus 0,1 --dataset cityscapes --train_type ontrain --train_data_list ./dataset/list/Cityscapes/cityscapes_train_list.txt --max_epochs 300
  ```
  
  2. training on train+val set
  ```
  python cityscapes_train.py --gpus 0,1 --dataset cityscapes --train_type ontrainval --train_data_list ./dataset/list/Cityscapes/cityscapes_trainval_list.txt --max_epochs 350
  ```
  3. Evaluation (on validation set)
 
  ```
  python cityscapes_eval.py --gpus 0 --val_data_list ./dataset/list/Cityscapes/cityscapes_val_list.txt --resume ./checkpoint/cityscapes/CGNet_M3N21bs16gpu2_ontrain/model_cityscapes_train_on_trainset.pth
  ```
  
  - model file download: [model_cityscapes_train_on_trainset.pth](https://pan.baidu.com/s/1rilPxLqBH57_sLg0Lc1--Q)
  
  4. Testing (on test set)
  ```
  python cityscapes_test.py --gpus 0 --test_data_list ./dataset/list/Cityscapes/cityscapes_test_list.txt --resume ./checkpoint/cityscapes/CGNet_M3N21bs16gpu2_ontrainval/model_cityscapes_train_on_trainvalset.pth
  ```
  - model file download: [model_cityscapes_train_on_trainvalset.pth](https://pan.baidu.com/s/1x7LEunjweoDvb_-xNQmFAg)
  5. Running time on Tesla V100 (single card single batch)
  ```
  56.8 ms with command "torch.cuda.synchronize()"
  20.0 ms w/o command "torch.cuda.synchronize()"
  ```
  
###  For Camvid
  1. training on train+val set
   ```
  python camvid_train.py
  ```
  2. testing (on test set)
  ```
  python camvid_test.py
  ```

  - model file download: [model_camvid_train_on_trainvalset.pth](https://pan.baidu.com/s/1gH6pI3jFmtlBgjgLUCjVvA)
  
  ## Citation
If CGNet is useful for your research, please consider citing:
```
  @article{wu2020cgnet,
  title={Cgnet: A light-weight context guided network for semantic segmentation},
  author={Wu, Tianyi and Tang, Sheng and Zhang, Rui and Cao, Juan and Zhang, Yongdong},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={1169--1179},
  year={2020},
  publisher={IEEE}
}
```
  ## License

This code is released under the MIT License. See [LICENSE](LICENSE) for additional details.

## Thanks to the Third Party Libs
https://github.com/speedinghzl/Pytorch-Deeplab.
