from tqdm import tqdm
import torch
import sys
from torch.nn.utils import prune
import torch.nn as nn
import torch.autograd.profiler as profiler

def train_one_epoch(model,trainloader,criterion,optimizer,epoch,device):
    ####create bar
    bar = tqdm(total=len(trainloader), desc="[Train]")

    ####initialize loss
    epoch_loss = 0

    #### learning process
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #scheduler.step()
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


def validate(model,validloader,criterion,epoch,device):
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


def ptrain(model,trainloader,validloader,criterion,optimizer,epochs,device,writer,name,overfitting,ratio,PATH):

    min_val_loss = 100000
    max_val_acc = 0
    end = 0

    PATH1 = 'logs/'+PATH+'/model_weights.pth'

    model.load_state_dict(torch.load(PATH1))

    for name1, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
            module = prune.l1_unstructured(module, 'weight', ratio)
            module = prune.remove(module, name='weight')



    for epoch in range(epochs):
        print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)

        model, training_loss = train_one_epoch(model,trainloader,criterion,optimizer,epoch,device)
        val_loss,val_acc = validate(model,validloader,criterion,epoch,device)

        #scheduler.step(val_loss)
        writer.add_scalars('Losses_pruning', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
        writer.add_scalar('Validation Accuracy pruning', val_acc  , epoch + 1)
        writer.flush()

        if overfitting == 'acc':
            if max_val_acc < val_acc:
                best_model = model
                #min_val_loss = val_loss
                max_val_acc = val_acc
                ## save model
                PATH = './logs/{}/model_weights_p{}.pth'.format(name,ratio)
                torch.save(model.state_dict(),PATH)
                end = epoch
                print('==> best model saved <==')
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))
            else:
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))

        else :
            if val_loss < min_val_loss and abs(val_loss - training_loss) < 0.2:
                best_model = model
                min_val_loss = val_loss
                ## save model
                PATH = './logs/{}/model_weights_p{}.pth'.format(name,ratio)
                torch.save(model.state_dict(),PATH)
                end = epoch
                print('==> best model saved <==')
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))
            else:
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))


    f= open("./logs/{}/epochs_overfitting.txt".format(name),"w+")
    f.write('epoch nb {} , val acc {} , val loss {}'.format(end +1, max_val_acc, min_val_loss))
    f.close()

def test(model,testloader,criterion,device,name,ratio) :

    #### pruning

    PATH1 = 'logs/'+name+'/model_weights.pth'.format(ratio)

    model.load_state_dict(torch.load(PATH1))

    #module = model.conv1

    C = []
    L = []

    for name1, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d)  :
            C.append((module,'weight'))
            #module = prune.l1_unstructured(module, 'weight', ratio )
        elif isinstance(module, torch.nn.Linear) :
            L.append((module,'weight'))
            #module = prune.l1_unstructured(module, 'weight', ratio)

    prune.global_unstructured(C[5:10],pruning_method=prune.L1Unstructured,amount=ratio)
    prune.global_unstructured(C[10:],pruning_method=prune.L1Unstructured,amount=ratio+0.4)
    prune.global_unstructured(L,pruning_method=prune.L1Unstructured,amount=ratio+0.2)
#        elif isinstance(module, torch.nn.Linear) :
#            module = prune.l1_unstructured(module, 'weight', ratio + 0.2)

            #module = prune.random_structured(module, 'weight', ratio,dim =0)
            #module = prune.ln_structured(module, 'weight', ratio,n =2,dim=0)
    for name1, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
            print(
                "Sparsity in {}t: {:.2f}%".format(name1,
                    100. * float(torch.sum(module.weight == 0))
                    / float(module.weight.nelement())
                )
            )
    sum = 0
    totals = 0
    for tuple in C + L :
        sum += torch.sum(tuple[0].weight == 0)
        totals += tuple[0].weight.nelement()

    print(
    "Global sparsity: {:.2f}%".format(sum/totals * 100)
    )
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
            f= open("./logs/{}/profiler.txt".format(name),"w+")
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
    f= open("./logs/{}/results_mixpruning_ratio{}.txt".format(name,ratio),"w+")
    f.write(' -> Test Accuracy = {}'.format(test_acc))
    print('\n ')
    f.write(' -> Test Loss     = {}'.format(test_loss))
    print('\n ')
    f.write(' -> global sparsity   = {}'.format(sum/totals))
    f.close()

    return test_loss,test_acc
