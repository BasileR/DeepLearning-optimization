from tqdm import tqdm
import torch


def train_one_epoch(bcmodel,trainloader,criterion,optimizer,epoch,device):
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
        optimizer.zero_grad(set_to_none = True)
        # forward + backward + optimize
        bcmodel.binarization()
        outputs = bcmodel.model(inputs)
        loss_step  = criterion(outputs, labels)
        loss_step.backward()
        bcmodel.restore()
        optimizer.step()
        bcmodel.clip()
        # print statistics
        running_loss = loss_step.item()
        epoch_loss+=running_loss
        bar.set_description("[Train] Loss = {:.4f}".format(round(running_loss, 4)))
        bar.update()

    epoch_loss = epoch_loss/len(trainloader)
    bar.close()

    return bcmodel,epoch_loss


def validate(bcmodel,validloader,criterion,epoch,device):
    bar = tqdm(total=len(validloader), desc="[Val]")
    val_loss = 0
    bcmodel.model.eval()
    total = 0
    correct = 0
    bcmodel.binarization()
    for i, data in enumerate(validloader,0):

        #extract data
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass but without grad

        with torch.no_grad():
            pred = bcmodel.model(inputs)

        # update loss, calculated by cpu
        val_loss += criterion(pred,labels).cpu().item()
        bar.set_description("[Val] Loss = {:.4f}".format(round(val_loss/len(validloader), 4)))
        bar.update()

        ## into tensorboard
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    bcmodel.restore()

    val_loss = val_loss/len(validloader)
    val_acc = correct/total



    bar.close()

    return val_loss,val_acc


def train(bcmodel,trainloader,validloader,criterion,optimizer,epochs,device,writer,name,overfitting):

    min_val_loss = 100000
    max_val_acc = 0
    end = 0

    for epoch in range(epochs):
        print('='*10 + ' epoch ' + str(epoch+1) + '/' + str(epochs) + ' ' + '='*10)

        model, training_loss = train_one_epoch(bcmodel,trainloader,criterion,optimizer,epoch,device)
        val_loss,val_acc = validate(bcmodel,validloader,criterion,epoch,device)
        #scheduler.step(val_loss)
        writer.add_scalars('Losses', {'val' : val_loss ,'train' : training_loss}  , epoch + 1)
        writer.add_scalar('Validation Accuracy', val_acc  , epoch + 1)
        writer.flush()

        if overfitting == 'acc':
            if max_val_acc < val_acc:
                best_model = model
                #min_val_loss = val_loss
                max_val_acc = val_acc
                min_val_loss = val_loss
                ## save model
                PATH = './logs/{}/model_weights.pth'.format(name)
                torch.save(bcmodel.model.state_dict(),PATH)
                end = epoch
                print('==> best model saved <==')
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))
            else:
                print('  -> Training   Loss     = {}'.format(training_loss))
                print('  -> Validation Loss     = {}'.format(val_loss))
                print('  -> Validation Accuracy = {}'.format(val_acc))

                end = epoch
        else :
            if val_loss < min_val_loss and abs(val_loss - training_loss) < 0.2:
                best_model = model
                min_val_loss = val_loss
                max_val_acc = val_acc
                ## save model
                PATH = './logs/{}/model_weights.pth'.format(name)
                torch.save(bcmodel.model.state_dict(),PATH)
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
    f.write('epoch nb {} , val acc {} , val loss {}'.format(end +1, min_val_loss, max_val_acc))
    f.close()


def test(bcmodel,testloader,criterion,device,PATH) :

    PATH1 = 'logs/'+PATH+'/model_weights.pth'

    bcmodel.model.load_state_dict(torch.load(PATH1))
    #### set bar
    bar = tqdm(total=len(testloader), desc="[Test]")

    #### set model to eval mode
    bcmodel.model.eval()
    bcmodel.binarization()

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
        with torch.no_grad():
            pred = bcmodel.model(inputs)

        # update loss, calculated by cpu
        running_loss = criterion(pred,labels).cpu().item()
        bar.set_description("[Test] Loss = {:.4f}".format(round(running_loss, 4)))
        bar.update()

        test_loss += criterion(pred,labels).cpu().item()

        _, predicted = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    bcmodel.restore()

    test_acc = 100 * correct / total
    test_loss = test_loss/len(testloader)

    ## save prediction
    bar.close()

    print(' -> Test Accuracy = {}'.format(test_acc))
    print(' -> Test Loss     = {}'.format(test_loss))
    f= open("./logs/{}/results_bin.txt".format(PATH),"w+")
    f.write(' -> Test Accuracy = {}'.format(test_acc))
    print('\n')
    f.write(' -> Test Loss     = {}'.format(test_loss))
    f.close()

    return test_loss,test_acc
