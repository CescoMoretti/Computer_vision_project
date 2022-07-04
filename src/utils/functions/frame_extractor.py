import argparse
import os
import cv2
import multiprocessing

# To make it work on vscode add this to your .vscode/launch.json and run in debug mode
# "args": [
#                 "--video_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/",
#                 "--out_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/"
#             ],

class Frame_extractor:
    def __init__(self, frames_dir):
        """
        function extracts frames from vidoe files

        video: mp4 file name of video
        frames_dir: path to frame output directory
        video_dir: path to directory containing mp4 videos
        """
        self.frame_dir = frames_dir
        self.out_dir = None
        self.video_path = None

    def load_video(self, video_path):
        #carica cartella output e creala se non esiste
        # video.split('.')[0].zfill(3)
        self.out_dir = os.path.join(self.frame_dir,"test", 'rgb')
        if not os.path.isdir(self.out_dir) or len(os.listdir(self.out_dir)) != 1800:
            os.makedirs(self.out_dir, exist_ok=True)

        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.count = 1
    
    def extract(self):
        for i in range(0,5):
            success, image = self.video_cap.read()
            self.count +=1
        self.video_cap.set(1, self.count-1)

        # filename = os.path.join(self.out_dir, str(self.count).zfill(4) + '.jpg')
        # print(filename)
        # cv2.imwrite(filename, image)
        
        return image