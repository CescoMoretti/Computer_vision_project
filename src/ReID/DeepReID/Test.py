# Cose da fare
# 1) Sistemare le varie cartelle
# 2) Capire cosa fa nn.Functional.interpolate (Riga 106)
# 3) Capire bene la funzione extract_feature
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ReIDMOdel
#from utils import fuse_all_conv_bn

###########################################################
### Option command line
### -------------------
###########################################################
def args():
    parser = argparse.ArgumentParser(description='test')

    #Model
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')

    #Data
    parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='Directory of the data for training')
    parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
    parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
    parser.add_argument('--h', default=256, type=int, help='Height of transformation')
    parser.add_argument('--w', default=128, type=int, help='width of transformation')
    parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()

    return opt

################################################################
### Load Data
### ---------
################################################################
def loadData(h = 256, w = 128, data_dir = "C:\Market-1501-v15.09.15\pytorch", batchsize=32):
    
    data_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform) for x in ['gallery', 'query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                  shuffle = False, num_workers=16) for x in ['gallery', 'query']}

    class_names = image_datasets['query'].classes

    return dataloaders, image_datasets, class_names

################################################################
### Load Network
### ------------
################################################################
def load_network(network, epoch):
    save_path = os.path.join('C:\Computer_vision_project\src\ReID\DeepReID\ModelResult', 'net_%s.pth'%epoch)
    network.load_state_dict(torch.load(save_path))
    return network

################################################################
### Extract Feature
### ---------------
################################################################
def fliplr(img):
    # Flip horizontal
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long() # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloader, batchsize, ms):
    count = 0
    linear_num = 2048

    for iter, data in enumerate(dataloader):
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, linear_num).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img)
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs

        #norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter==0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])
        start = iter*batchsize
        end = min((iter+1)*batchsize, len(dataloader.dataset))
        features[start:end,:] = ff
    return features


################################################################
### Get_id
### ------
################################################################
def get_id(img_path):
    camera_id = []
    labels = []

    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

################################################################
### Test
### ----
################################################################
def test(image_datasets, dataloaders, nclasses, epoch, name, batchsize, ms):
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    print('-----------test------------')

    model_structure = ReIDMOdel(class_num=nclasses)

    model = load_network(model_structure, epoch)

    # Remove the final fc layer and classifier layer
    model.classifier = nn.Sequential()

    #Change to test mode
    model = model.eval()

    #print(model)

    #Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'], batchsize=batchsize, ms=ms)
        query_feature = extract_feature(model, dataloaders['query'], batchsize=batchsize, ms=ms)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed//60, time_elapsed%60))
    
    #Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label':gallery_label, 'gallery_cam': gallery_cam, 'query_f': query_feature.numpy(), 'query_label':query_label, 'query_cam':query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    print(name)
    result = 'C:\Computer_vision_project\src\ReID\DeepReID\TestReID\\result.txt'
    os.system('python evaluate_gpu.py | tee -a %s'%result)


################################################################
### Main
### ----
################################################################
if __name__=="__main__":
    
    # Argument by command line
    opt = args()
    use_gpu = opt.gpu_ids
    name = opt.name
    epoch = opt.which_epoch
    test_dir = opt.test_dir
    batchsize = opt.batchsize
    h = opt.h
    w = opt.w
    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # Configuration of train model
    config_path = os.path.join("C:\Computer_vision_project\src\ReID\DeepReID\ModelResult", 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    if 'nclasses' in config:
        nclasses = config['nclasses']
    else:
        nclasses=751

    dataloaders, image_datasets, class_names = loadData(h=h, w=w, data_dir="C:\Market-1501-v15.09.15\pytorch", batchsize=batchsize)

    test(image_datasets=image_datasets, dataloaders=dataloaders, nclasses=nclasses, epoch=epoch, name=name, batchsize=batchsize, ms=ms)
