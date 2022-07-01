<!-- exp1 step lr
exp2 cosin lr
shufflenet_v2_x1_0 warmup + cosin lr
exp4 warmup + cosin lr + warmup-epochs 3
  python3 main.py --arch shufflenet_v2_x1_0 --workers 8 --epochs 100 --batch 64 --warmup-epochs 3  --name exp4 ../data/mini-imagenet/ -->
# Introduction
The knowledge distilling of shufflenet_v2 and resnet model on mini-imagenet dataset.

# Quick Start
## dataset
Download [mini-imagenet](https://drive.google.com/drive/folders/1Td3qB6WCJMzKMOb-YdTict5302S8RqF1?usp=sharing), put it in ../data/ directory and unzip it.

## install
```
git clone https://github.com/wangziren1/knowledge_distill_on_mini-imagenet.git
cd knowledge_distill_on_mini-imagenet
pip install -r requirements.txt
```
## train

1. train single model:
```
python3 main.py --arch shufflenet_v2_x1_0 --workers 8 --epochs 100 --batch 64 --warmup-epochs 1 --name shufflenet_v2_x1_0 ../data/mini-imagenet/
```
--name: save directory  

2. knownledge distill from big model to small model: 
```
python3 kd.py --arch shufflenet_v2_x1_0 --big shufflenet_v2_x2_0 --workers 8 --epochs 100 --batch 64 --warmup-epochs 1 --name kd_T3_alpha0.9 --T 3 --alpha 0.9 ../data/mini-imagenet/
```
--arch: small model  
--big: big model  
--name: save directory  
--T: temperature  
--alpha: the weight of softloss, totalloss = alpha*softloss + (1-alpha)*hardloss     

# Results
## tricks
We do some tricks experiments on shufflenet_v2_x1_0 model.
| tricks | accuracy |
| :---:  | :---:  |
| steplr | 76.71 |
| cosinlr | 77.72 |
| cosinlr + warmup 1 epoch| **78.84** |
| cosinlr + warmup 3 epoch| 77.76 |

In the next experiments, we will use the "cosinlr + warmup 1 epoch" setting.
## shufflenet
| shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 |
| :---:              | :---:              | :---:              |
| 78.84              |79.1                | 80.83              |

Distill the knowledge in shufflenet_v2_x2_0 to shufflenet_v2_x1_0. T represents temperature and alpha represents the weight of softloss(totalloss = alpha*softloss + (1-alpha)*hardloss).
| alpah\T |    1  |  3     | 5     | 10    |
| :---:   | :---:| :---: | :---: | :---: |
| 0.9     | 79.64 | 81.10 | 80.65 | 80.875 |
| 0.7     |       | 81.27 |       |        |
| 0.6     |       | **81.28** |       |        |
| 0.5     |       | 80.78 |       |        |

When T = 3 and alpha = 0.6, we get the hightest accuracy. In the resnet experiment, we will use this setting.
## resnet
| resnet18 | resnet34 | resnet50 |
| :---:    | :---:    | :---:    |
| 79.11    | 81       | 81.79    |

Distill the knowledge in resnet50 to resnet18  
| alpah\T |    3 |
| :---:   | :---:|
| 0.6     | 82.45 |
# Some findings
1. The knowledge distilling can improve accurary of small model compared to that of small model and even big model.
2. The knowledge distilling can increase generalization: the decrease of the gap between train accuracy and val accuracy. For example, the train accuracy and the val accuracy of shufflenet_x1 are 86.46 and 78.84 which differs by 7.62, while the train accuracy and the val accuracy of distilled shufflenet_x1 are 87.28 and 81.28 which differs by 6. The train accuracy and the val accuracy of resnet18 are 93.28 and 79.11 which differs by 14.17, while the train accuracy and the val accuracy of distilled resnet18 are 92.53 and 82.45 which differs by 10.08.

# Reference
* https://zhuanlan.zhihu.com/p/102038521
* https://blog.csdn.net/qq_37541097/article/details/113027489
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/mini_imagenet
* https://github.com/pytorch/examples/blob/main/imagenet/main.py