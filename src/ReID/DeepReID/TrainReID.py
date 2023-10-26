############ STEP COMMIT
# 3) Sistemare le variabili per la GPU
# 4) Sistemare l'optimizer

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
from model import ReIDMOdel
import yaml
from shutil import copyfile
version =  torch.__version__
from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning

##############################################################################
### Options command line
### --------------------
##############################################################################
def args():
    parser = argparse.ArgumentParser(description='Training')

    #Model
    parser.add_argument('--use_gpu', default='0', type=str, help='Id of gpu if avaible')
    parser.add_argument('--name', default='ResNet50', type=str, help='Name of the nerual network (default: ResNet 50)')

    #Data
    parser.add_argument('--data_dir', default='C:/Market-1501-v15.09.15/pytorch', type=str, help='Directory of the data for training')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--h', default=256, type=int, help='Height of transformation')
    parser.add_argument('--w', default=128, type=int, help='width of transformation')
    parser.add_argument('--n_workers', default=8, type=int, help='Number of workers for dataloaders')

    #Optimizer
    parser.add_argument('--lr', default=0.05, type=int, help='Learning Rate')
    parser.add_argument('--total_epoch', default=15, type=int, help='Total epoch for training')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')

    #Loss
    parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0')

    opt = parser.parse_args()
    cudnn.enabled = True
    cudnn.benchmark = True


    return opt

###########################################
### Load of the dataset
### ------------------
def get_data(dataset_path='../Market/pytorch', train_all=False, batchsize=32, h=256, w=128, n_workers=8):
    
    transfom_train_list = [
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform = transforms.Compose(transfom_train_list)

    train_all = ''
    if train_all:
        train_all = '_all'

    image_dataset = {}
    image_dataset['train'] = datasets.ImageFolder(os.path.join(dataset_path, 'train' + train_all),
                                                  transform)
    image_dataset['val'] = datasets.ImageFolder(os.path.join(dataset_path, 'val'),
                                                transform)
    
    dataloaders = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=batchsize,
                                                  shuffle=True, num_workers=n_workers, pin_memory=True,
                                                  prefetch_factor = 2, persistent_workers = True)
                                                for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_dataset[x]) for x in ['train', 'val']}
    class_names = image_dataset['train'].classes

    return dataloaders, dataset_sizes, class_names


##############################################################################
### Train model
### -----------
##############################################################################
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, bathcsize=32, num_epochs = 25):
    gpu = True
    y_loss = {} #loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        #Each epoch has a train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True) #Set model in training mode
            else:
                model.train(False) #Set model in evaluation mode

            running_loss = 0.0
            running_corrects = 0.0

            #Iterate over data
            count = 0
            for (data, dlabel) in dataloaders[phase]:
                #Get the input
                inputs, labels = data, dlabel
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size<bathcsize:
                    continue
                if gpu:
                    inputs, labels = Variable(inputs.cuda().detach()), Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                #Forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                del inputs

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if int(version[0]) > 0 or int(version[2]) > 3:
                    running_loss += loss.item() * now_batch_size
                else:
                    running_loss += loss.data[0] * now_batch_size
                del loss
                running_corrects += float(torch.sum(preds==labels.data))

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1-epoch_acc)

            if phase=='val' and epoch%4==0:
                last_model_wts = model.state_dict()
                if(gpu):
                    save_network(model.module, epoch+1)
                else:
                    save_network(model, epoch+1)

            if phase == 'val':
                draw_curve(epoch, y_loss, y_err)
            if phase == 'train':
                scheduler.step()
        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed%60))
        print()

    model.load_state_dict(last_model_wts)
    if(gpu):
        save_network(model.module, 'last')
    else:
        save_network(model, 'last')

    return model


##############################################################################
### Draw the curve
### --------------
##############################################################################
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, y_loss, y_err):
    x_epoch.append(epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('.\ModelResult','train.jpg'))

##############################################################################
### Save the network
### ----------------
##############################################################################
def save_network(model, epoch):
    save_filename = 'net_%s.pth'% epoch
    save_path = os.path.join('.\ModelResult',save_filename)
    torch.save(model.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        model.cuda(0)
    pass

##############################################################################
### Main
### ----
##############################################################################

if __name__ == '__main__':
    
    opt = args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("You are running on", torch.cuda.get_device_name(), "gpu.")

    dataloaders, dataset_sizes, class_names = get_data(opt.data_dir, opt.train_all, opt.batchsize, opt.h, opt.w, opt.n_workers)

    model = ReIDMOdel(len(class_names))
    
    #print(model)

    optim_name = optim.SGD
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        ignored_params = list(map(id, model.module.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        classifier_params = model.module.classifier.parameters()
        optimizer_ft = optim_name([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        classifier_params = model.classifier.parameters()
        optimizer_ft = optim_name([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)

    dir_name = ".\ModelResult"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with open('%s\opts.yaml'%dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    criterion = nn.CrossEntropyLoss().to(device=device)
    print(device)
    model = train_model(model=model, criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes, device=device, bathcsize=opt.batchsize, num_epochs=opt.total_epoch)