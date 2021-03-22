# ResNet-optim

## Introduction

Hi ! Here is our repository for the AI optimization course at IMT Atlantique. The goal of this course was fitting a neural network for cifar10 and performing quantization and pruning to **minimize the micronet score** of the model. Enjoy !


You can find here the repository of the course : https://github.com/brain-bzh/ai-optim . This repository is also inspired by https://github.com/kuangliu/pytorch-cifar and https://github.com/eghouti/BinaryConnect. Check out these great repos ! 

## Prerequisites 

This code is written in Ptyhon 3.8 and we used PyTorch 1.7.0. 

Other packages :
Numpy, Tensorboard, Torchvision, tqdm, matplotlib and argparse. Please make sure that these packages are installed. 

## Structure of the repository

This repo is made of :
- the tp3_main.py script which contains the train, test and other useful functions to prune the network
- resnet.py and densenet.py which are the scripts of the ResNet and DenseNet models (see  https://github.com/kuangliu/pytorch-cifar)
- profiler.py to compute micronet score (see https://github.com/brain-bzh/ai-optim)
- minicifar.py : create a small cifar10 with 4 classes and its dataloaders (see https://github.com/brain-bzh/ai-optim)
- binaryconnect.py and tp3_bin.py to perform neural network binarization (see https://github.com/eghouti/BinaryConnect)
- logs folder will be created as we used tensorboard writer

Do not hesitate to look intot the code, it is well commented.

## How to use this repository

Training, test and pruning processes can be easily done directly from the terminal using options: 

| Option | Type | Description | Default value |
|:--------:|:------:|:-------------:|:---------------:|
|  name  | str  | name of the experiment | demo |
|  score  | bool  | compute micronet score | False |
|  modelToUse  | str  | name of the architecture to use (resnets or densenets)| ResNet18 |
|  dataset  | str  | name of the dataset to use (minicifar, cifar10 or cifar100)| minicifar |
|  train  | bool  | perform training if selected| False |
|  test  | bool  | perform test if selected| False |
|  ptrain  | bool  | perform training and pruning iteratively | False |
|  optimizer  | str  | optimizer to use (adam or sgd) | sgd |
|  lr  | float  | learning rate for optimizer | 0.01 |
|  momentum  | float  | momentum for optimizer| 0.9 |
|  decay  | float  | weight decay for optimizer| 5e-4 |
|  epochs  | int  | number of epochs for training| 300 |
|  batch_size  | int  | size of the batch to load | 32 |
|  overfitting  | str  | function to optimize to save best model (accuracy or loss) | loss |
|  path  | str  | name of the folder where are the pretrained weights to load | None |
|  pruning  | bool  | perform pruning on the model | False |
|  method  | str  | perform pruning on the model (uniform or global) | global |
|  ratio  | float  | ratio of parameters to prune and total number of parameters | 0.3 |


### Examples of commands

- To train a model on cifar10 :

```
python tp3_main.py --train --dataset cifar10 --overfitting accuracy --name cifar10_resnet18 --modelToUse ResNet18 --lr 0.01 --momentum 0.9 --decay 5e-4 --batch_size 32 --epochs 300
```

- To test a model on cifar10 :

```
python tp3_main.py --test --dataset cifar10  --modelToUse ResNet18 --batch_size 32 --path  cifar10_resnet18
```

- To prune a model and test it on cifar10 :

```
python tp3_main.py --test --dataset cifar10  --modelToUse resnet18 --batch_size 32 --path cifar10_resnet18 --pruning --method global --ratio 0.45
```

- To prune a model and get its micronet score on cifar10 :

```
python tp3_main.py --dataset cifar10  --modelToUse resnet18 --path cifar10_resnet18 --pruning --method global --ratio 0.45 --score
```



## Our experiments

### Micronet Score and goal of the course

The Micronet Challenge (https://micronet-challenge.github.io/) was about creating the lightest network which has a high accuracy on ImageNet, cifar100 or WikiText-103. Models were evaluated considering a micronet score, that was made of two parts : one to evaluate the number of computations required, and the other one to evaluate the number of parameters.

The objective of this project was taking a existing network and reduce it as much as possible to **minimize its micronet score while keeping the accuracy over 90 %**.

### Baseline and first reduction of the model

We decided first to work on Resnet18 (https://arxiv.org/abs/1512.03385, implementation here : https://github.com/kuangliu/pytorch-cifar) with 12M parameters. We trained it from scratch on cifar10 and obtain 92.1% of accuracy. So as to decrease the number of parameters, we divided by 4 the number for feature maps created by each convolution : We got a model with 700k parameters , accuracy = 90.42 % and micronet score = 0.1260. 



