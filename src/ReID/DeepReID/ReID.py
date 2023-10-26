import argparse
import torch
from torchvision.ops import roi_pool
import os
from ReID.DeepReID.model import ReIDMOdel
from ReID.DeepReID.PeopleDB import PeopleDB
import torchvision.transforms as T
from ReID.DeepReID.distances import L2_distance

class ReID():
    
    def __init__(self, path_model, h = 256, w = 128, epoch = "13"):

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

    def resize_bb(self, b_bbs):
        n = len(b_bbs)
        count = 0

        for i in range(n):
            bb = torch.Tensor(b_bbs[i]).view(1, 4)
            zero = torch.zeros((1,1))
            bb = torch.cat((zero, bb), 1)

            if count == 0:
                cords = bb
            else:
                cords = torch.cat((cords, bb))
            count = count + 1

        return cords

    #LOAD NETWORK
    def load_network(self, network, epoch, path_model):
        save_path = os.path.join(path_model, 'net_%s.pth'%epoch)
        network.load_state_dict(torch.load(save_path))
        return network
    
    #ANALYZE
    def analyze(self, frame, b_bbs):

        #Vedere formato bounding box

        frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0) / 255.0
        b_bbs = self.resize_bb(b_bbs)
        crops = roi_pool(frame_tensor, b_bbs, output_size=[self.h, self.w])
        crops=self.transform(crops)

        target_ids = []
        feature_vectors = self.model(crops)
        for i in range(feature_vectors.shape[0]):
            vector = feature_vectors[i]
            target_id, new = self.db.Get_ID(vector)
            target_ids.append(target_id)

        self.db.Update_Frame()

        return target_ids