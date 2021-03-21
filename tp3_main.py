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
parser.add_argument('--name', type = str , default = 'demo', help = 'name of the experience' )
parser.add_argument('--score', action='store_true' , default = False, help = 'micronet score')

## model used
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
    '''
    Parameters :
    ------------
    dataset (str) : name of the dataset to use
    batch_size (int) : size of the batchs for dataloaders
    modelToUse (str) : name of the model to used

    Returns :
    ---------
    model : wanted model
    train/valid/testloaders : dataloaders for training, validation and test
    '''
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

    '''
    Parameters :
    ------------
    model (object) : model that will be pruned
    pruning_method (str) : method of pruning that will be used
                           global     : see pytorch global_unstructured function
                           uniform    : see pytorch prune function, applied to each layer
    ratio (float) : ratio for pruning

    Returns :
    ---------
    model : pruned model with a weight_mask (not remove)
            For more information, see how pruning is done in pytorch and remove function
    '''

    if pruning_method == 'uniform':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
                module = prune.l1_unstructured(module, 'weight', ratio)

    elif pruning_method == 'global':

        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d)  :
                parameters_to_prune.append((module,'weight'))

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio)

    return model

def get_sparsity(model):
    '''
    Parameters :
    ------------
    model (object) : model that will be pruned

    Prints :
    ---------
    Percentage of zeros in each layer
    '''
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

def pos_zeros(model):
    '''
    Parameters :
    ------------
    model (object) : model that will be pruned

    Prints :
    ---------
    Number of feature maps that have more than 50% of zeros for each layer
    '''
    L = []

    for name1, module in model.named_modules():
        zeros = []
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            L.append((module,'weight'))
            for i in range(module.weight_mask.shape[0]):
                A = int(100*float(torch.sum(module.weight_mask[i] == 0))/float(module.weight_mask[i].nelement()))
                zeros.append(A)
            print('{0:20} {1} / {2}'.format(name1, len([x for x in zeros if x > 50.0]),len(zeros)))

def train_one_epoch(model,trainloader,criterion,optimizer,epoch):
    '''
    Description :
    ------------
    Perform training of the model for one epoch

    Parameters :
    ------------
    model (object) : model to train
    trainloader (object) : Dataloader of training set
    criterion (object) : loss to use for training (see pytorch documentation)
    optimizer (object) : optimizer to use for training (see pytorch documentation)

    Returns :
    ---------
    model : model trained for one epoch
    epoch_loss (float) : mean loss of the epoch
    '''
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
    '''
    Description :
    ------------
    Perform validation of the model

    Parameters :
    ------------
    model (object) : model to train
    validloader (object) : Dataloader of validation set
    criterion (object) : loss to use for validation (see pytorch documentation)
    optimizer (object) : optimizer to use for validation (see pytorch documentation)

    Returns :
    ---------
    val_acc (float) : mean accuracy of the validation
    val_loss (float) : mean loss of the validation
    '''
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

    '''
    Description :
    ------------
    Perform training and validation (at each epoch) for several epochs

    Parameters :
    ------------
    model (object) : model to train
    trainloader (object) : Dataloader of training set
    validloader (object) : Dataloader of validation set
    criterion (object) : loss to use for validation (see pytorch documentation)
    optimizer (object) : optimizer to use for validation (see pytorch documentation)
    epochs (int) : number of epochs for training
    name (str) : name of the experience

    Returns :
    ---------
    val_acc (float) : mean accuracy of the validation
    val_loss (float) : mean loss of the validation
    '''

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

def test(model,testloader,criterion,device) :
    '''
    Description :
    ------------
    Perform test of the model

    Parameters :
    ------------
    model (object) : model to test
    testloader (object) : Dataloader of test set
    criterion (object) : loss to use for test (see pytorch documentation)
    optimizer (object) : optimizer to use for test (see pytorch documentation)

    Returns :
    ---------
    test_acc (float) : mean accuracy of the test
    test_loss (float) : mean loss of the test
    '''
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

    bar.close()

    print(' -> Test Accuracy = {}'.format(test_acc))
    print(' -> Test Loss     = {}'.format(test_loss))


    return test_loss,test_acc

def load_weights(model,PATH):
    '''
    Description :
    ------------
    Load weights on model, with weight mask or not

    Parameters :
    ------------
    model (object) : model where to load wieghts
    path (str) : folder to find .pth file
    Returns :
    ---------
    model (model) : model with loaded weights
    '''

    PATH = 'logs/'+args.path+'/model_weights.pth'
    state_dict = torch.load(PATH)

    ## check if weight_mask and create if needed
    if 'conv1.weight_mask' in state_dict.keys():
        for name, module in backbonemodel.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d):
                module = prune.identity(module, 'weight')

    model.load_state_dict(state_dict)

    return model

def get_micronet_score(model,pruning,method,ratio):
    '''
    Description :
    ------------
    Print micronet score and it computation. Kill process after getting it

    Parameters :
    ------------
    model (object) : model to evaluate
    pruning (bool) : True if pruning is required
    method (str)   : method of pruning
    ratio (float)  : ratio of pruning
    Prints :
    ---------
    Micronet score
    '''
    if pruning:
        backbonemodel = get_prune_model(model,method,ratio)
    score = profiler.main(model)
    sys.exit('Kill after getting micronet score : {}'.format(score))

def get_nb_params(model):
    return sum(p.numel() for p in model.parameters())

## get model and dataloaders

backbonemodel , trainloader , validloader , testloader = get_model_dataset(args.dataset,args.batch_size,args.modelToUse)


##### check number of parameters ####
params = get_nb_params(backbonemodel)


#### print and save experince config ####
print('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
print('{0:20} {1}'.format('model', args.modelToUse))
print('{0:20} {1}'.format('Nb of parameters',params))

for arg in vars(args):
    print('{0:20} {1}'.format(arg, getattr(args, arg)))
print('{0:20} {1}'.format('GPU',use_gpu))
print('='*10 + '==================' + '='*10)




#### create optimizer, criterion and scheduler ####

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(backbonemodel.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay)
#optimizer = optim.Adam(backbonemodel.parameters(), lr=args.lr,weight_decay=args.decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)



## get pretrained weights if path is not none
if args.path != None :
    backbonemodel = load_weights(backbonemodel,args.path)


## if pruning is selected, then prune model and print its sparsity
if args.pruning:
    backbonemodel = get_prune_model(backbonemodel,args.method,args.ratio)
    get_sparsity(backbonemodel)


## if --score is selected, then calculate the micronet score of the model
if args.score :
    get_micronet_score(backbonemodel,args.pruning,args.method,args.ratio)
## training and test processes

if args.train :
    ## create tensorboard writer
    writer = SummaryWriter('logs/'+args.name)

    ## save config in a text file
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

    ## load model in the device (cpu or cuda)
    backbonemodel = backbonemodel.to(device)

    ## train function
    train(backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name)


### ptrain is training and pruning iteratively
elif args.ptrain :

    ## load the model in the gpu or cpu
    backbonemodel = backbonemodel.to(device)

    ## first validation to see the first model
    val_loss,val_acc = validate(backbonemodel,validloader,criterion,0)
    print(' == First validation before training == ' )
    print('  -> Validation Loss     = {}'.format(val_loss))
    print('  -> Validation Accuracy = {}'.format(val_acc))

    ## training and pruning processes
    ratio = args.ratio
    for i in range(3):
        print(' = '*10 )

        ## increase pruning ratio
        dratio = 0.20
        ratio += (1-ratio)*dratio

        ## prune the model
        backbonemodel = get_prune_model(backbonemodel,args.method, dratio)

        ## tensorboard writer and training process
        writer = SummaryWriter('logs/'+args.name + str(ratio))
        train(backbonemodel,trainloader,validloader,criterion,optimizer,args.epochs,args.name + str(ratio))

## test process
if args.test :
    ## .half to divide by 2 the precision on weights
    backbonemodel = backbonemodel.half().to(device)
    ##test process
    test_loss, test_acc = test(backbonemodel,testloader,criterion,device)
    utils.save_test_results(args.path,test_acc,test_loss,args.pruning,args.ratio)

else:
    sys.exit('Need to select either --train or --test or --ptrain')
