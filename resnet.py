'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


def train_one_epoch(self,trainloader,criterion,optimizer,epoch):
    '''
    Description :
    ------------
    Perform training of the model for one epoch

    Parameters :
    ------------
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
        outputs = self(inputs)
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

    return self,epoch_loss

def validate(self,validloader,criterion,epoch):
    '''
    Description :
    ------------
    Perform validation of the model

    Parameters :
    ------------
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
    self.eval()
    total = 0
    correct = 0
    for i, data in enumerate(validloader,0):

        #extract data
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass but without grad

        with torch.no_grad():
            pred = self(inputs)

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes,power_in_planes):
        super(ResNet, self).__init__()
        self.in_planes = 2**power_in_planes

        self.conv1 = nn.Conv2d(3, 2**power_in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2**power_in_planes)
        print(num_blocks)
        if num_blocks[0] != 0:
            self.layer1 = self._make_layer(block, 2**power_in_planes, num_blocks[0], stride=1)
        if num_blocks[1] != 0:
            self.layer2 = self._make_layer(block, 2**(power_in_planes+1), num_blocks[1], stride=2)
        if num_blocks[2] != 0:
            self.layer3 = self._make_layer(block, 2**(power_in_planes+2), num_blocks[2], stride=2)
        if num_blocks[3] != 0:
            self.layer4 = self._make_layer(block, 2**(power_in_planes+3), num_blocks[3], stride=2)
        self.linear = nn.Linear(2**(power_in_planes+4)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def train(self,trainloader,validloader,criterion,optimizer,epochs,name):

        '''
        Description :
        ------------
        Perform training and validation (at each epoch) for several epochs

        Parameters :
        ------------
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
        ## create tensorboard writer
        writer = SummaryWriter('logs/'+name)

        min_val_loss = 100000
        max_val_acc = 0
        end = 0

        for epoch in range(epochs):

            print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)
            self, training_loss = train_one_epoch(self,trainloader,criterion,optimizer,epoch)
            val_loss,val_acc = validate(self,validloader,criterion,epoch)
            scheduler.step()
            writer.add_scalars('Losses', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
            writer.add_scalar('Validation Accuracy', val_acc  , epoch + 1)
            writer.flush()

            if args.overfitting == 'accuracy':
                if max_val_acc < val_acc :
                    best_model = self
                    max_val_acc = val_acc
                    ## save model

                    utils.save_weights(best_model,name)
                    end = epoch
                    print('==> best model saved <==')
                    utils.save_train_results(name,val_acc,val_loss,end+1)
            elif args.overfitting == 'loss':
                if val_loss < min_val_loss and abs(val_loss-training_loss) < 0.2 :
                    best_model = self
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


def ResNet18(N, num_blocks ,power_in_planes):
    return ResNet(BasicBlock,num_blocks = num_blocks, num_classes = N, power_in_planes = 4)


def ResNet34(N, num_blocks ,power_in_planes):
    return ResNet(BasicBlock,num_blocks =num_blocks , num_classes = N, power_in_planes = 6)



def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
