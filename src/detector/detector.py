import sys, os
sys.path.append(os.getcwd()+"/src/utils/") 
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions.frame_extractor import Frame_extractor

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class Detector():

    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.classes = self.model.names
        self.progressiveId = 0

    def detect(self, frame):
        result = self.model.predict(frame, device=0, classes=[0])[0] #this is to pass detection of the first frame (we only give one)
        return result.boxes.cpu().numpy().cls, result.boxes.cpu().numpy().conf, result.boxes.cpu().numpy().xyxy

    def draw_resut(self, frame, classIds, scores, boxes, color):
        for (classid, score, box) in zip( classIds, scores, boxes):
                label = "%s : %f" % (self.classes[classid], score)
                cv2.rectangle(frame, np.asarray(box[:2], dtype = 'int'), np.asarray(box[2:4], dtype = 'int'), color, 2)
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Detection result", frame)
        cv2.waitKey(1)