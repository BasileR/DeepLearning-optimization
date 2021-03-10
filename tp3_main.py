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
from torch.nn.utils import prune
##dataset
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
##libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
##scripts
import resnet
import densnet
import binaryconnect
import tp3_bin
import utils
import profiler



#### check GPU usage ####

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

#### add parser ####

parser = argparse.ArgumentParser()

## infos
parser.add_argument('--name', type = str ,default = 'demo' )
parser.add_argument('--score', action='store_true' , default = False, help = 'micronet score')

##model utilisé
parser.add_argument('--modelToUse', type = str, default = 'ResNet18' , choices = ['ResNet18','ResNet34','ResNet50','ResNet101','ResNet152', 'DensNet121', 'DensNet169', 'DensNet201', 'DensNet161', 'DensNetCifar'], help ='Choose ResNet model to use')

## dataset
parser.add_argument('--dataset', type = str , choices = ['minicifar','cifar10','cifar100'] , default = 'minicifar' )

## training settings
parser.add_argument('--train', action='store_true' , default = False, help = 'perform training')
parser.add_argument('--ptrain', action='store_true' , default = False, help = 'perform iterative/pruning training')
parser.add_argument('--lr', type = float, default = 1e-2 , help = 'Learning rate')
parser.add_argument('--momentum', type = float, default = 0.9 , help = 'momentum for Learning Rate')
parser.add_argument('--decay', type = float, default = 5e-4 , help = 'decay')
parser.add_argument('--epochs', type = int, default = 300  , help = 'Number of epochs for training')
parser.add_argument('--batch_size', type = int, default = 32 , help ='Batch size for DataLoader')
parser.add_argument('--overfitting', type = str , default = 'loss' , choices = ['loss','accuracy'], help ='Choose overfitting type')
parser.add_argument('--optimizer', type = str , choices = ['sgd','adam'], default = 'sgd' )

## quantization
parser.add_argument('--test', action='store_true' , default = False, help = 'perform test')
parser.add_argument('--path', type = str , help = 'path to pth in desired logs to find model_weights')
parser.add_argument('--pruning', action='store_true' , default = False, help = 'perform pruning' )
parser.add_argument('--method',  type = str , choices = ['uniform','global','decreasing'], default = 'global' )
parser.add_argument('--ratio', type = float, default = 0.3 , help = 'ratio for pruning')

args = parser.parse_args()

#### choose dataset and set dataloaders ####

def get_model_dataset(dataset,batch_size,modelToUse):
    if dataset == 'minicifar':

        trainloader = DataLoader(minicifar_train,batch_size=batch_size,sampler=train_sampler)
        validloader = DataLoader(minicifar_train,batch_size=batch_size,sampler=valid_sampler)
        testloader = DataLoader(minicifar_test,batch_size=batch_size)
        n = 4

    elif dataset == 'cifar10':

        ## add data augmentation
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
        trainloader = DataLoader(dataset,batch_size=batch_size)
        validloader = DataLoader(test_dataset,batch_size=batch_size)
        testloader = DataLoader(test_dataset,batch_size=batch_size)
        n = 10

    elif dataset == 'cifar100':

        transform_train_1 = [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]

        transform_test_1 = [transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]

        transform_train = transforms.Compose(transform_train_1)
        transform_test = transforms.Compose(transform_test_1)

        dataset = CIFAR100(root='data/', download=True, transform=transform_train)
        test_dataset = CIFAR100(root='data/', train=False, transform=transform_test)

        val_size = 10000
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        trainloader = DataLoader(dataset,batch_size=batch_size)
        validloader = DataLoader(test_dataset,batch_size=batch_size)
        testloader = DataLoader(test_dataset,batch_size=batch_size)
        n = 100


    if modelToUse == 'ResNet18' :
        model = resnet.ResNet18(N=n)

    elif modelToUse == 'ResNet34' :
        model = resnet.ResNet34(N=n)

    elif modelToUse == 'ResNet50' :
        model = resnet.ResNet50(N=n)

    elif modelToUse == 'ResNet101' :
        model = resnet.ResNet101(N=n)

    elif modelToUse == 'ResNet152' :
        model = resnet.ResNet152(N=n)

    elif modelToUse == 'DensNet121':
        model = densnet.DenseNet121(N=n)

    elif modelToUse == 'DensNet169':
        model = densnet.DenseNet169(N=n)

    elif modelToUse == 'DensNet201':
        model = densnet.DenseNet201(N=n)

    elif modelToUse == 'DensNet161':
        model = densnet.DenseNet161(N=n)

    elif modelToUse == 'DensNetCifar':
        model = densnet.densenet_cifar(N=n)


    return model , trainloader , validloader , testloader

def get_prune_model(model,pruning_method,ratio):

    if pruning_method == 'uniform':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
                module = prune.l1_unstructured(module, 'weight', ratio)
                #module = prune.ln_structured(module, 'weight', ratio,dim = 1 , n = 2)


    elif pruning_method == 'global':

        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d)  :
                parameters_to_prune.append((module,'weight'))

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
        )

    return model

def get_sparsity(model):

    L = []
    for name1, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d)  :
            L.append((module,'weight'))
            txt1 = "Sparsity in {}t: {:.2f}%".format(name1,
                100. * float(torch.sum(module.weight == 0))
                / float(module.weight.nelement()))
            print(txt1)

    sum = 0
    totals = 0
    for tuple in L :
        sum += torch.sum(tuple[0].weight == 0)
        totals += tuple[0].weight.nelement()
        txt = "Global sparsity: {:.2f}%".format(sum/totals * 100)
    print(txt)

def train_one_epoch(model,trainloader,criterion,optimizer,epoch):
    ####create bar
    bar = tqdm(total=len(trainloader), desc="[Train]")

    ####initialize loss
    epoch_loss = 0

    #### learning process
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss_step  = criterion(outputs, labels)
        loss_step.backward()
        optimizer.step()
        # print statistics
        running_loss = loss_step.item()
        epoch_loss+=running_loss
        bar.set_description("[Train] Loss = {:.4f}".format(round(running_loss, 4)))
        bar.update()

    epoch_loss = epoch_loss/len(trainloader)
    bar.close()

    return model,epoch_loss

def validate(model,validloader,criterion,epoch):
    bar = tqdm(total=len(validloader), desc="[Val]")
    val_loss = 0
    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(validloader,0):

        #extract data
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass but without grad

        with torch.no_grad():
            pred = model(inputs)

        # update loss, calculated by cpu

        loss = criterion(pred,labels).cpu().item()
        val_loss += loss
        bar.set_description("[Val] Loss = {:.4f}".format(round(loss, 4)))
        bar.update()

        ## into tensorboard
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss = val_loss/len(validloader)
    val_acc = correct/total

    bar.close()

    return val_loss,val_acc

def train(model,trainloader,validloader,criterion,optimizer,epochs,name):

    min_val_loss = 100000
    max_val_acc = 0
    end = 0

    for epoch in range(epochs):

        print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)
        model, training_loss = train_one_epoch(model,trainloader,criterion,optimizer,epoch)
        val_loss,val_acc = validate(model,validloader,criterion,epoch)
        scheduler.step()
        writer.add_scalars('Losses', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
        writer.add_scalar('Validation Accuracy', val_acc  , epoch + 1)
        writer.flush()

        if args.overfitting == 'accuracy':
            if max_val_acc < val_acc :
                best_model = model
                max_val_acc = val_acc
                ## save model

                utils.save_weights(model,name)
                end = epoch
                print('==> best model saved <==')
                utils.save_train_results(name,val_acc,val_loss,end+1)
        elif args.overfitting == 'loss':
            if val_loss < min_val_loss and abs(val_loss-training_loss) < 0.2 :
                best_model = model
                min_val_loss = val_loss
                ## save model
                utils.save_weights(model,name)
                end = epoch
                print('==> best model saved <==')
                utils.save_train_results(name,val_acc,val_loss,end+1)
        print('  -> Training   Loss     = {}'.format(training_loss))
        print('  -> Validation Loss     = {}'.format(val_loss))
        print('  -> Validation Accuracy = {}'.format(val_acc))

def test(model,testloader,criterion,device,PATH) :

    PATH1 = 'logs/'+PATH+'/model_weights.pth'

    model.load_state_dict(torch.load(PATH1))

    if args.pruning:
        model = get_prune_model(model,args.method,args.ratio)
        get_sparsity(model)
    #### set bar
    bar = tqdm(total=len(testloader), desc="[Test]")

    #### set model to eval mode
    model.eval()

    total = 0
    correct = 0
    test_loss = 0

    for i, data in enumerate(testloader):

        #extract data
        inputs, labels = data

        inputs = inputs.half().to(device)
        labels = labels.to(device)

        # set  loss
        running_loss = 0

        # forward pass but without grad
        with torch.no_grad():
            pred = model(inputs)


        # update loss, calculated by cpu
        running_loss = criterion(pred,labels).cpu().item()
        bar.set_description("[Test] Loss = {:.4f}".format(round(running_loss, 4)))
        bar.update()

        test_loss += criterion(pred,labels).cpu().item()

        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_loss = test_loss/len(testloader)



    ## save prediction
    utils.save_test_results(PATH,test_acc,test_loss)
    bar.close()

    print(' -> Test Accuracy = {}'.format(test_acc))
    print(' -> Test Loss     = {}'.format(test_loss))


    return test_loss,test_acc


backbonemodel , trainloader , validloader , testloader = get_model_dataset(args.dataset,args.batch_size,args.modelToUse)

if args.score :
    if args.pruning:
        backbonemodel = get_prune_model(backbonemodel,args.method,args.ratio)
    score = profiler.main(backbonemodel)
    sys.exit('Kill after getting micronet score')



#### create optimizer, criterion and scheduler ####

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(backbonemodel.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay)
#optimizer = optim.Adam(backbonemodel.parameters(), lr=args.lr,weight_decay=args.decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)


##### check number of parameters ####
params = sum(p.numel() for p in backbonemodel.parameters())


#### print and save experince config ####
print('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
print('{0:20} {1}'.format('model', 'ResNet18'))
print('{0:20} {1}'.format('Nb of parameters',params))

for arg in vars(args):
    print('{0:20} {1}'.format(arg, getattr(args, arg)))
print('{0:20} {1}'.format('GPU',use_gpu))
print('='*10 + '==================' + '='*10)

#### training and test processes ####


if args.train :

    if args.path != None :
        PATH = 'logs/'+args.path+'/model_weights.pth'
        backbonemodel.load_state_dict(torch.load(PATH))

    if args.pruning:
        backbonemodel = get_prune_model(backbonemodel,args.method,args.ratio)
        get_sparsity(backbonemodel)

    writer = SummaryWriter('logs/'+args.name)
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

    backbonemodel = backbonemodel.to(device)
    train(backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name)

elif args.ptrain :

    for name, module in backbonemodel.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d):
            module = prune.identity(module, 'weight')

    PATH = 'logs/'+args.path+'/model_weights.pth'
    backbonemodel = backbonemodel.to(device)
    backbonemodel.load_state_dict(torch.load(PATH))

    val_loss,val_acc = validate(backbonemodel,validloader,criterion,0)
    print(' == First validation before training == ' )
    print('  -> Validation Loss     = {}'.format(val_loss))
    print('  -> Validation Accuracy = {}'.format(val_acc))

    get_sparsity(backbonemodel)
    ratio = args.ratio

    for i in range(3):
        print(' ================ ' )
        dratio = 0.20
        ratio += (1-ratio)*dratio
        backbonemodel = get_prune_model(backbonemodel,args.method, dratio)
        get_sparsity(backbonemodel)
        writer = SummaryWriter('logs/'+args.name + str(ratio))
        train(backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name + str(ratio))




if args.test :
    backbonemodel = backbonemodel.half().to(device)
    test(backbonemodel,testloader,criterion,device,args.path)

else:
    print('Need to select either --train or --test')
