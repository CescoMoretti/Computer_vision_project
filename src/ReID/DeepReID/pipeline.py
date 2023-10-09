import argparse
import torch
import numpy as np
from torchvision.ops import roi_pool
import time
import os
from model import ReIDMOdel
from PeopleDB import PeopleDB
import torchvision.transforms as T
import cv2
import random
from distances import L2_distance
#from utils import fuse_all_conv_bn

def args():
    parser = argparse.ArgumentParser(description='test')

    #Model
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--path_model', default="ModelResult", type=str, help="Path of the model to load")

    #Data
    parser.add_argument('--video', default="C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/Motsynth/MOTSynth_1/000.mp4", type=str)
    parser.add_argument('--annotation', default="C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/MOT/mot_annotations/000/gt/gt.txt", type=str)

    #Transform
    parser.add_argument('--h', default=256, type=int, help='Height of transformation')
    parser.add_argument('--w', default=128, type=int, help='width of transformation')

    opt = parser.parse_args()

    return opt

# Load annotation
def load(filepath):
    data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        print("Oooops, cant parse %s, skipping this one..." , filepath)

    return data

################################################################
### Load Network
### ------------
################################################################
def load_network(network, epoch, path_model):
    save_path = os.path.join(path_model, 'net_%s.pth'%epoch)
    network.load_state_dict(torch.load(save_path))
    return network

################################################################
### Extract bounding box
### --------------------
################################################################
def extract_bb(data, frame, f, transform):
    count = 0
    thisF = np.flatnonzero(data[:,0]==f)
    for i in thisF:
        x1 = ((data[i, 2]-1)).astype(int)
        y1 = ((data[i, 3]-1)).astype(int)
        width = ((data[i, 4])).astype(int)
        heigth = ((data[i, 5])).astype(int)

        x2 = x1 + width
        y2 = y1 + heigth

        bb = torch.Tensor([x1, y1, x2, y2]).view(1, 4)
        zero = torch.zeros((1,1))
        bb = torch.cat((zero, bb), 1)

        if count == 0:
            cords = bb
        else:
            cords = torch.cat((cords, bb))

        count = count + 1

    return cords

##################################################
### Plotting
### --------
##################################################
def plot_history(history, frame, color, line_thickness=3):
    for i in range(len(history)-1):
        cv2.line(frame, history[i], history[i+1], color, line_thickness)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #print(f"c1:{c1}, c2:{c2}")
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



#Analizzare video per riedintificazione
def analyze(name_video, name_annotation, path_model, transform, h = 256, w = 128, epoch = "last"):
    
    #Aprire i relativi file e caricare il modello
    video = cv2.VideoCapture(name_video)
    max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    data = load(name_annotation)
    model_structure = ReIDMOdel()

    model = load_network(model_structure, epoch, path_model)

    model = model.eval()

    db = PeopleDB(dist_function=L2_distance, dist_threshold=1.0, frame_memory=20*10, max_descr_per_id=3)
    ncolors = 255
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(ncolors)]

    f = 1

    while f < max_frame - 10:
        print("Frame " + str(f))
        video.set(1, f)
        ret, frame = video.read()
        if not ret:
            print("Impossible read video")
            break
        else:
            pass

        bbs = extract_bb(data, frame, f, transform)
        #print(bbs)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        crops = roi_pool(frame_tensor, bbs, output_size = [h, w])
        crops = transform(crops)
        #print("Crops shape: " + str(crops.shape))
        #print(crops)
        target_ids = []
        feature_vectors = model(crops)
        print(feature_vectors.shape)

        for i in range(feature_vectors.shape[0]):
            vector = feature_vectors[i]
            target_id, new = db.Get_ID(vector)
            target_ids.append(target_id)
        db.Update_Frame()

        for i in range(len(bbs)):
            id = int(target_ids[i])

            mean = (int((bbs[i,1]+bbs[i,3])/2), int((bbs[i,2]+bbs[i,4])/2))

            plot_one_box(bbs[i, 1:5], frame, label=str(id), color=colors[id % ncolors], line_thickness=2)
            plot_history(db.Update_ID_position(id, mean), frame, color=colors[id % ncolors],line_thickness=2)

        cv2.imshow("img", frame)
        cv2.waitKey(0)


        f = f + 10


if __name__ == "__main__":
    
    opt = args()
    name_model = opt.name
    epoch = opt.which_epoch
    name_video = opt.video
    name_annotation = opt.annotation
    path_model = opt.path_model
    h = opt.h
    w = opt.w

    transform = T.Compose([
        #T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    analyze(name_video, name_annotation, path_model, transform, h, w, epoch)

