from ultralytics import YOLO
from ultralytics import settings
import torch
import os

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    torch.cuda.empty_cache()
    model = YOLO('yolov8n.pt')
    model.train(data='./train.yaml',
                epochs=150, 
                imgsz=544, 
                cache= True,
                #device="cpu", 
                workers=0,
                batch=4, 
                #resume="False"
                )


if __name__=="__main__": 
    main() 