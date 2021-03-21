# ResNet-optim

## Introduction

Hi ! Here is our repository for the AI optimization course at IMT Atlantique. The goal of this course was fitting a neural network for cifar10 and performing quantization and pruning to minimize the micronet score of the model. You can find here the repository of the course : https://github.com/brain-bzh/ai-optim . This repository is also inspired by https://github.com/kuangliu/pytorch-cifar and https://github.com/eghouti/BinaryConnect. Check out these great repos ! 

## Structure of the repository

This repo is made of :
- the tp3_main.py script which contains the train, test and other useful fonctions for pruning the network.
- resnet.py and densenet.py whiwh are the scripts of the ResNet and DenseNet models
- logs folder will be created as we used tensorboard writer

## How to use this repo

Training, test and pruning processes can be easily done directly from the terminal using options: 

| Option | Type | Description | Default |
|--------|------|-------------|---------|




## Baseline first reduction of the model

We decided first to work on Resnet18 (implementation here : https://github.com/kuangliu/pytorch-cifar) with 12M parameters. We trained it from scratch on cifar10 and obtain 92.1% of accuracy. So as to decrease the number of parameters, we divided by 4 the number for feature maps created by each convolution : We got a model with 700k parameters , accuracy = 90.42 % and micronet score = 0.1260. 



