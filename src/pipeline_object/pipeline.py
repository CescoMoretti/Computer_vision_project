from detector.detector import Detector
from detector.detector_yv8 import Detector_yv8
import sys, os
sys.path.append(os.getcwd()+"/src/utils/") 
from functions.frame_extractor import Frame_extractor
from shapely.geometry import Polygon
import cv2

this_path = os.getcwd()
class Pipeline():
    def __init__(self, video_dir):
        self.this_path = os.getcwd()
        self.people_detector = Detector('yolov8n.pt')
        self.head_detector = Detector(this_path + '/src/detector/yolov8/best.pt')
        self.frame_extractor = Frame_extractor()
        self.frame_extractor.load_video(video_dir)
        self.frame_extractor.get_width()
        self.width = self.frame_extractor.get_width()
        self.height = self.frame_extractor.get_width()


    def process_frame(self):
        frame, finished_video = self.frame_extractor.extract()
        people_classIds, people_scores, people_boxes = self.people_detector.detect(frame)
        head_classIds, head_scores, head_boxes = self.head_detector.detect(frame)
        for h_box in head_boxes:
            best_person_box = [0,0,0,0]
            best_person_box_iou = 0
            for p_box in people_boxes:
                iou = Pipeline.calculate_iou(h_box, p_box)
                if iou > 0.08: 
                    best_person_box = p_box
                    best_person_box_iou = iou
            if best_person_box_iou != 0.0:
                self.people_detector.draw_resut(frame, people_classIds, people_scores, [best_person_box], cv2.COLOR_BGR2HSV)
                self.head_detector.draw_resut(frame, head_classIds, head_scores, [h_box], cv2.COLOR_LUV2BGR) 
                #histogram = histogram_finder.getHistogram(bouniding_box)
                #distance = distance_finder.findDistance(bouniding_box) -->creare oggetto per estrarre teste
                #histogram finder / salva dati riconoscimento
                #distance finderer
       
        # input("Press Enter to continue...")

        
    def calculate_iou(box_1, box_2):
        poly_2 = Polygon([[box_1[0] , box_1[1]], #ad
                          [box_1[2] , box_1[1]], #as
                          [box_1[2] , box_1[3]], #bs
                          [box_1[0] , box_1[3]]]) #bd
        
        poly_1 = Polygon([[box_2[0] , box_2[1]],
                          [box_2[2] , box_2[1]],
                          [box_2[2] , box_2[3]],
                          [box_2[0] , box_2[3]]])
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou