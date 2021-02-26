#### Imports ####
##torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms
##dataset
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torchvision.datasets import CIFAR10
##libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
##scripts
import resnet
import binaryconnect
import tp3_none
import tp3_bin
import tp3_halftest
import tp3_pruning
import utils
#### check GPU usage ####

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

#### add parser ####

parser = argparse.ArgumentParser()

## infos
parser.add_argument('--name', type = str ,default = 'demo' )
parser.add_argument('--test', action='store_true' , default = False, help = 'perform test')
parser.add_argument('--train', action='store_true' , default = False, help = 'perform training')
parser.add_argument('--path', type = str , help = 'path to pth in desired logs to find model_weights')

## dataset
parser.add_argument('--dataset', type = str , choices = ['minicifar','cifar10','cifar100'] , default = 'minicifar' )

## training settings
parser.add_argument('--lr', type = float, default = 1e-2 , help = 'Learning rate')
parser.add_argument('--momentum', type = float, default = 0.9 , help = 'momentum for Learning Rate')
parser.add_argument('--decay', type = float, default = 1e-5 , help = 'decay')
parser.add_argument('--epochs', type = int, default = 150  , help = 'Number of epochs for training')
parser.add_argument('--scheduler', action='store_true' , default = True, help = 'add a "Cosine" with factor 0.1"')
parser.add_argument('--factor', type = float, default = 1e-1 , help = 'ReduceLROnPlateau factor')
parser.add_argument('--batch_size', type = int, default = 32, help ='Batch size for DataLoader')
parser.add_argument('--overfitting', type = str , default = 'loss' ,choices = ['loss','accuracy'], help ='Choose overfitting type')

## optimizer
parser.add_argument('--optimizer', type = str , choices = ['sgd','adam'], default = 'sgd' )

## quantization
parser.add_argument('--quantization', type = str , choices = ['half','binarization','pruning','none'] , default = 'none' )
parser.add_argument('--ratio', type = float, default = 0.3 , help = 'ratio for pruning')

args = parser.parse_args()

#### choose dataset and set dataloaders ####

if args.dataset == 'minicifar':
    trainloader = DataLoader(minicifar_train,batch_size=args.batch_size,sampler=train_sampler)
    validloader = DataLoader(minicifar_train,batch_size=args.batch_size,sampler=valid_sampler)
    testloader = DataLoader(minicifar_test,batch_size=args.batch_size)
    #### create model ####
    backbonemodel = resnet.ResNet18(N = 4)
elif args.dataset == 'cifar10':

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset = CIFAR10(root='data/', download=True, transform=transform_train)
    test_dataset = CIFAR10(root='data/', train=False, transform=transform_test)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    trainloader = DataLoader(dataset,batch_size=args.batch_size)
    validloader = DataLoader(test_dataset,batch_size=args.batch_size)
    testloader = DataLoader(test_dataset,batch_size=args.batch_size)
    #### create model ####
    backbonemodel = resnet.ResNet18(N = 10)

#### add tensorboard writer ####
if args.train :
    writer = SummaryWriter('logs/'+args.name)

#### create optimizer, criterion and scheduler ####

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(backbonemodel.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


##### check number of parameters ####
params = sum(p.numel() for p in backbonemodel.parameters())

def train(script,model,trainloader,validloader,criterion,optimizer,epochs,name):

    min_val_loss = 100000
    max_val_acc = 0
    end = 0

    for epoch in range(epochs):

        print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)
        model, training_loss = script.train_one_epoch(model,trainloader,criterion,optimizer,epoch,device)
        val_loss,val_acc = script.validate(model,validloader,criterion,epoch,device)
        scheduler.step()
        writer.add_scalars('Losses', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
        writer.add_scalar('Validation Accuracy', val_acc  , epoch + 1)
        writer.flush()

        if args.overfitting == 'accuracy':
            if max_val_acc < val_acc :
                best_model = model
                max_val_acc = val_acc
                ## save model
                if script == tp3_bin:
                    utils.save_weights(model.model,name)
                else:
                    utils.save_weights(model,name)
                end = epoch
                print('==> best model saved <==')
                utils.save_train_results(name,val_acc,val_loss,end+1)
        elif args.overfitting == 'loss':
            if val_loss < min_val_loss and abs(val_loss-training_loss) < 0.2 :
                best_model = model
                min_val_loss = val_loss
                ## save model
                if script == tp3_bin:
                    utils.save_weights(model.model,name)
                else:
                    utils.save_weights(model,name)
                end = epoch
                print('==> best model saved <==')
                utils.save_train_results(name,val_acc,val_loss,end+1)
        print('  -> Training   Loss     = {}'.format(training_loss))
        print('  -> Validation Loss     = {}'.format(val_loss))
        print('  -> Validation Accuracy = {}'.format(val_acc))






#### print and save experince config ####
print('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
print('{0:20} {1}'.format('model', 'ResNet18'))
print('{0:20} {1}'.format('Nb of parameters',params))

for arg in vars(args):
    print('{0:20} {1}'.format(arg, getattr(args, arg)))
print('{0:20} {1}'.format('GPU',use_gpu))
print('='*10 + '==================' + '='*10)

if args.train :
    f = open('./logs/{}/experience_config.txt'.format(args.name),'w+')
    f.write('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
    f.write('\n')
    for arg in vars(args):
        f.write('{0:20} {1}'.format(arg, getattr(args, arg)))
        f.write('\n')
    f.write('{0:20} {1}'.format('GPU',use_gpu))
    f.write('\n')
    f.write('{0:20} {1}'.format('Nb of parameters',params))
    f.write('\n')
    f.write('='*10 + '==================' + '='*10)
    f.close()

#### training and test processes ####

if args.quantization == 'binarization':
    script = tp3_bin
    backbonemodel = binaryconnect.BC(backbonemodel)
    backbonemodel.model = backbonemodel.model.to(device)
else:
    if args.quantization == 'half':
        backbonemodel = backbonemodel.half()
        script = tp3_half
    if args.quantization == 'pruning':
        script = tp3_pruning
    if args.quantization == 'none':
        script = tp3_none
    backbonemodel = backbonemodel.to(device)

if args.train and not args.test :
     train(script,backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name)

elif args.train and args.test:
    train(script,backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name)
    script.test(backbonemodel,testloader,criterion,device,args.path)

elif args.test:
    if args.quantization =='half':
        backbonemodel = backbonemodel.to(device)
        tp3_halftest.test(backbonemodel,testloader,criterion,device,args.path)
    elif args.quantization =='binarization':
        bcmodel = binaryconnect.BC(backbonemodel)
        bcmodel.model = bcmodel.model.to(device)
        tp3_bin.test(bcmodel,testloader,criterion,device,args.path)
    elif args.quantization =='none':
        backbonemodel = backbonemodel.to(device)
        tp3_none.test(backbonemodel,testloader,criterion,device,args.path)
    elif args.quantization =='pruning':
        backbonemodel = backbonemodel.to(device)
        tp3_pruning.test(backbonemodel,testloader,criterion,device,args.name,args.ratio)

else:
    print('Need to select either --train or --test')
