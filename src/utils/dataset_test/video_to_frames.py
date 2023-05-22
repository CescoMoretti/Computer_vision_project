import argparse
import os
import sys
import cv2
import multiprocessing
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.functions.frame_extractor import Frame_extractor

# To make it work on vscode add this to your .vscode/launch.json and run in debug mode
# "args": [
#                 "--video_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/",
#                 "--out_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/"
#             ],

def parse_video(video, out_dir, video_dir):
    frame_extractor = Frame_extractor()
    frame_extractor.save_all_frame_from_video(out_dir, video, video_dir)
def main():
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument('--video_dir', help='Directory hosting videos', required=True)
    parser.add_argument('--out_dir', help='Directory of output', required=True)

    args = parser.parse_args()

    video_dir = args.video_dir
    out_dir = args.out_dir

    video_list = os.listdir(video_dir)
    pool = multiprocessing.Pool(processes=2)
    print("start: out=", out_dir,"\n input= ", video_dir)

    #[pool.apply_async(frame_extractor.extract, (video, frames_dir, video_dir)) for video in video_list]
    [pool.apply_async(parse_video, (video, out_dir, video_dir)) for video in video_list]
    pool.close()
    pool.join()




if __name__ == '__main__':
    main()
