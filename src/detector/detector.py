import sys, os
sys.path.append(os.getcwd()+"/src/utils/") 

import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions.frame_extractor import Frame_extractor

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class Detector():

    def __init__(self, weights_path, cfg_path, l_names_path):
        self.frame_extractor = Frame_extractor()

        self.this_path = os.getcwd()
        self.weights_path = weights_path
        self.cfg_path = cfg_path
        self.l_names_path = l_names_path

        with open(self.l_names_path, 'r') as f:
            self.classes = f.read().splitlines()
        self.yolo_net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.model = cv2.dnn_DetectionModel(self.yolo_net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

        
        self.progressiveId = 0

    def inference_from_video(self, video_list):
        #function to take a list of videos and make inference frame by frame
        for video in video_list:
            frame, video_finished = self.frame_extractor.extract()
            if(video_finished):
                return None, None, None
            classIds, scores, boxes = self.detect(frame)
            return classIds, scores, boxes


    def detect(self, frame):
        classIds, scores, boxes = self.model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)        
        return classIds, scores, boxes

    def draw_resut(self, frame, classIds, scores, boxes):
        for (classid, score, box) in zip( classIds, scores, boxes):
            if classid == 0: #TODO togliere appena sistemo la rete
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (self.classes[classid[0]], score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("test", frame)
        cv2.waitKey(1)