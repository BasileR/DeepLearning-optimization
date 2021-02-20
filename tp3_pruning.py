from tqdm import tqdm
import torch
import sys
from torch.nn.utils import prune
import torch.nn as nn
import torch.autograd.profiler as profiler

def test(model,testloader,criterion,device,PATH,ratio) :

    #### pruning

    PATH1 = 'logs/'+PATH+'/model_weights.pth'

    model.load_state_dict(torch.load(PATH1))

    #module = model.conv1

    L = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module = prune.l1_unstructured(module, 'weight', ratio)

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

        inputs = inputs.to(device)
        labels = labels.to(device)

        # set  loss
        running_loss = 0

        # forward pass but without grad
        if i == 0 :

            with profiler.profile(profile_memory = True, record_shapes = True, use_cuda = True) as prof:

                with torch.no_grad():
                    pred = model(inputs)
            print()
            f= open("./logs/{}/profiler.txt".format(PATH),"w+")
            f.write(prof.key_averages().table())
            f.close()
        else :
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
    bar.close()

    print(' -> Test Accuracy = {}'.format(test_acc))
    print(' -> Test Loss     = {}'.format(test_loss))
    f= open("./logs/{}/results_pruning_ratio{}.txt".format(PATH,ratio),"w+")
    f.write(' -> Test Accuracy = {}'.format(test_acc))
    print('\n ')
    f.write(' -> Test Loss     = {}'.format(test_loss))
    f.close()

    return test_loss,test_acc
