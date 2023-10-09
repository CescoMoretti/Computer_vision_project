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

class ReID():
    
    def __init__(self, path_model, h = 256, w = 128, epoch = "last"):

        #PARAMETER
        self.epoch = epoch
        self.h = h
        self.w = w

        #LOAD MODEL
        model_structur = ReIDMOdel()
        self.model = self.load_network(model_structur, epoch, path_model)
        self.model = self.model.eval()

        #CREATE DATABASE
        self.db = PeopleDB(dist_function=L2_distance, dist_threshold=1.0, frame_memory=20*10, max_descr_per_id=3)

        #TRANSFORM
        self.transform = T.Compose([
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
        ])

        return


    #LOAD NETWORK
    def load_network(self, network, epoch, path_model):
        save_path = os.path.join(path_model, 'net_%s.pth'%epoch)
        network.load_state_dict(torch.load(save_path))
        return network
    
    #ANALYZE
    def analyze(self, frame, h_bbs, b_bbs):

        #Vedere formato bounding box

        frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0) / 255.0
        crops = roi_pool(frame_tensor, b_bbs, output_size=[self.h, self.w])
        crops=self.transform(crops)

        target_ids = []
        feature_vectors = self.model(crops)
        for i in range(feature_vectors.shape[0]):
            vector = feature_vectors[i]
            target_id, new = self.db.Get_ID(vector)
            target_ids.append(target_id)

        self.db.Update_Frame()

        return