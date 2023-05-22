import os
import cv2
import time
from detector import Detector

this_path = os.getcwd()


def main():
    person_detector = Detector(this_path+'/src/detector/YOLOv4/yolov4_coco.weights',
                        this_path+'/src/detector/YOLOv4/yolov4.cfg',
                        this_path+'/src/detector/YOLOv4/person.names')

    impath = this_path+'/test_imgs/raw_images/mov_001_007587.jpg'    
    frame = cv2.imread(impath)

    classIds, scores, boxes = person_detector.detect(frame)
    person_detector.draw_resut(frame, classIds, scores, boxes)


if __name__ == "__main__":
    main()
