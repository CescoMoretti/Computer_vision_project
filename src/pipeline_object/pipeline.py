from detector.detector import Detector
import sys, os
sys.path.append(os.getcwd()+"/src/utils/") 
from functions.frame_extractor import Frame_extractor
from shapely.geometry import Polygon


this_path = os.getcwd()
class Pipeline():
    def __init__(self, out_dir, video_dir):
        self.this_path = os.getcwd()
        self.people_detector = Detector(this_path+'/src/detector/YOLOv4/yolov4_coco.weights',
                        this_path+'/src/detector/YOLOv4/yolov4.cfg',
                        this_path+'/src/detector/YOLOv4/person.names')
        self.head_detector = Detector(this_path+'/src/detector/YOLOv4/yolov4-head_last.weights',
                        this_path+'/src/detector/YOLOv4/yolov4-head.cfg',
                        this_path+'/src/detector/YOLOv4/head.names')
        self.frame_extractor = Frame_extractor()
        self.frame_extractor.load_video(video_dir)


    def process_frame(self):
        frame, finished_video = self.frame_extractor.extract()
        people_classIds, people_scores, people_boxes = self.people_detector.detect(frame)
        
        head_classIds, head_scores, head_boxes = self.head_detector.detect(frame)
        for h_box in head_boxes:
            for p_box in people_boxes:
                iou = Pipeline.calculate_iou(h_box, p_box)
                print(iou)
                if iou > 0.05:
                    self.people_detector.draw_resut(frame, people_classIds, people_scores, [p_box]) #TODO remove, it's for debug
                    self.head_detector.draw_resut(frame, head_classIds, head_scores, [h_box]) 
                    #histogram = histogram_finder.getHistogram(bouniding_box)
                    #distance = distance_finder.findDistance(bouniding_box) -->creare oggetto per estrarre teste
                    #histogram finder / salva dati riconoscimento
                    #distance finderer
                    
    def calculate_iou(box_1, box_2):
        poly_1 = Polygon([[box_1[0] , box_1[1]],
                          [box_1[0]+box_1[2] , box_1[1]],
                          [box_1[0]+box_1[2] , box_1[1]+box_1[3]],
                          [box_1[0] , box_1[1]+box_1[3]]])
        
        poly_2 = Polygon([[box_2[0] , box_2[1]],
                          [box_2[0]+box_2[2] , box_2[1]],
                          [box_2[0]+box_2[2] , box_2[1]+box_2[3]],
                          [box_2[0] , box_2[1]+box_2[3]]])
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou