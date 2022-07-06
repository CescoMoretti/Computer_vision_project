from people_detection.detector import People_detector
from utils.functions.frame_extractor import Frame_extractor
import os



class Pipeline():
    def __init__(self):
        self.this_path = os.getcwd()
        self.detector = People_detector(self.this_path+'/src/people_detection/YOLOv3/yolov3.weights',
                        self.this_path+'/src/people_detection/YOLOv3/yolov3.cfg',
                        self.this_path+'/src/people_detection/YOLOv3/coco.names')
        self.frame_extractor = Frame_extractor('c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/')
        self.frame_extractor.load_video("C:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/000.mp4")

    def process_frame(self):
        frame = self.frame_extractor.extract()
        id_img, img_path= self.detector.detectPeople(frame)
        
        #bouniding_box = detector.findpeople(img)
        #histogram = histogram_finder.getHistogram(bouniding_box)
        #distance = distance_finder.findDistance(bouniding_box) -->creare oggetto per estrarre teste
        #histogram finder / salva dati riconoscimento
        #distance findere