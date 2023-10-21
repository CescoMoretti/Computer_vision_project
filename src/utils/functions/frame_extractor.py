import argparse
from genericpath import isfile
import os
import cv2 
import multiprocessing
import path

# To make it work on vscode add this to your .vscode/launch.json and run in debug mode
# "args": [
#                 "--video_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/",
#                 "--out_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/"
#             ],

class Frame_extractor:
    def __init__(self):
        """
        function extracts frames from video files

        video: mp4 file name of video
        frames_dir: path to frame output directory
        video_dir: path to directory containing mp4 videos
        """
        self.max_frames = None
        self.out_dir = None
        self.video_path = None
        self.finished_video = False
        self.count = 0

    def load_video(self, video_path):
        if not os.path.isfile(video_path):
            print("ERROR: video path invalid")
            return -1

        
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.count = 0
        self.max_frames = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.finished_video = False

    def save_all_frame_from_video(self, frame_dir, video, video_dir):
        video_path = os.path.join(video_dir, video)
        self.load_video(video_path)
        #carica cartella output e creala se non esiste 
        self.out_dir = frame_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        if(os.path.isfile(os.path.join(self.out_dir, str(video).split(".")[0] + '_1800.jpeg'))):
            print("video " + str(video).split(".")[0] + " already processed")
            return
        
        while(not self.finished_video):
            filename = str(video).split(".")[0] + "_" + str(self.count).zfill(4)
            if(not os.path.isfile(os.path.join(self.out_dir, filename + '.jpeg'))):
                self.save_frame(filename)
            else:
                print(filename + " alredy present")
                self.video_cap.set(1, self.count+1)
                self.count += 1
        return
    
    def setOutir(self, out_dir):
        self.out_dir = out_dir


    def save_frame(self, filename):
        if self.out_dir == None:
            print("Error: select an out dir with setOutDir")
        file_path = os.path.join(self.out_dir, filename + '.jpeg')        
        image, finished_video = self.extract(1)
        print(file_path)
        if not os.path.isfile(file_path):
            cv2.imwrite(file_path, image)
        return image, self.finished_video


    def extract(self,skip_n_frame=1):
        if self.video_path == None:
            print("ERROR: load video before extracting, use load_video()")
            return -1

        if not self.finished_video:
            for i in range(0,skip_n_frame): #verify the next n frame are valid
                if self.count > self.max_frames:
                    self.finished_video = True
                    return None, self.finished_video
                success, image = self.video_cap.read()
                self.count +=1
            self.video_cap.set(1, self.count-1) #set the next frame to the last checked
        else:
            print("INFO: video finished")
        return image, self.finished_video
    def get_width(self):
        return self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    def get_height(self):
        return self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    
