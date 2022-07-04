from people_detection.detector import Detector
from utils.functions.frame_extractor import Frame_extractor
import os
import cv2 as cv

this_path = os.getcwd()

def main():

    detector = Detector(this_path+'/src/people_detection/YOLOv3/yolov3.weights',
                        this_path+'/src/people_detection/YOLOv3/yolov3.cfg',
                        this_path+'/src/people_detection/YOLOv3/coco.names')
    #histogram finder / salva dati riconoscimento
    #distance findere

    #prende frame img 
    frame_extractor = Frame_extractor('c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/')

    frame_extractor.load_video("C:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/000.mp4")
    frame = frame_extractor.extract()
    id_img, img_path= detector.detectPeople(frame)

    frame = frame_extractor.extract()
    id_img, img_path= detector.detectPeople(frame)

    frame = frame_extractor.extract()
    id_img, img_path= detector.detectPeople(frame)
    #bouniding_box = detector.findpeople(img)
    #histogram = histogram_finder.getHistogram(bouniding_box)
    #distance = distance_finder.findDistance(bouniding_box) -->creare oggetto per estrarre teste

if __name__ == "__main__":
    main()