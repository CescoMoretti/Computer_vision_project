from detector.detector import Detector
from ReID.DeepReID.ReID import ReID
from distance_estimator.DistanceEstimator import DistanceEstimator
import sys, os
sys.path.append(os.getcwd()+"/src/utils/") 
from shapely.geometry import Polygon
import cv2

this_path = os.getcwd()
class Pipeline():
    def __init__(self):
        self.this_path = os.getcwd()
        self.people_detector = Detector('yolov8n.pt')
        self.head_detector = Detector(this_path + '/src/detector/yolov8/best.pt')
        self.reid = ReID(this_path+ "/src/ReID/DeepReID/ModelResult/")
        self.distance_estimator = DistanceEstimator()

    def process_frame(self,frame):  
                
        #detections
        people_classIds, people_scores, people_boxes = self.people_detector.detect(frame)
        head_classIds, head_scores, head_boxes = self.head_detector.detect(frame)
    
        #matching bb and addind them to the lists
        bb_head_list = []
        bb_person_list = []
        for p_box in people_boxes:
            best_head_box = [0,0,0,0]
            best_head_box_iou = 0
            for h_box in head_boxes:
                iou = Pipeline.calculate_iou(h_box, p_box)
                if iou > 0.08: 
                    best_head_box = h_box
                    best_head_box_iou = iou
            if best_head_box_iou != 0.0:
                bb_head_list.append(best_head_box)
                bb_person_list.append(p_box)
                # self.people_detector.draw_resut(frame, people_classIds, people_scores, [p_box], cv2.COLOR_BGR2HSV)
                # self.head_detector.draw_resut(frame, head_classIds, head_scores, [best_head_box], cv2.COLOR_LUV2BGR) 
        print(len(bb_person_list)) 
        identification = self.reid.analyze(frame, bb_person_list)
        self.people_detector.draw_resut(frame, people_classIds, identification, bb_person_list, cv2.COLOR_BGR2HSV)

        # self.head_detector.draw_resut(frame, head_classIds, identification, bb_head_list, cv2.COLOR_LUV2BGR) 
            
        # distances = self.distance_estimator.computeDistance(identification, bb_person_list, bb_head_list)
        input("Press Enter to continue...")

        
    def calculate_iou(box_1, box_2):
        poly_2 = Polygon([[box_1[0] , box_1[1]],  #ad
                          [box_1[2] , box_1[1]],  #as
                          [box_1[2] , box_1[3]],  #bs
                          [box_1[0] , box_1[3]]]) #bd
        
        poly_1 = Polygon([[box_2[0] , box_2[1]],
                          [box_2[2] , box_2[1]],
                          [box_2[2] , box_2[3]],
                          [box_2[0] , box_2[3]]])
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou