import argparse
import os
import cv2
import multiprocessing
from utils.functions.frame_extractor import Frame_extractor

# To make it work on vscode add this to your .vscode/launch.json and run in debug mode
# "args": [
#                 "--video_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/",
#                 "--out_dir", "c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/"
#             ],


def main():
    frame_extractor = Frame_extractor('c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/test_out/')
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument(
        '--video_dir', help='Directory hosting videos', required=True)
    parser.add_argument('--out_dir', help='Directory of output', required=True)

    args = parser.parse_args()

    video_dir = args.video_dir
    out_dir = args.out_dir

    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    video_list = os.listdir(video_dir)
    pool = multiprocessing.Pool(processes=1)
    print("start", out_dir, video_dir)
    [pool.apply_async(frame_extractor.extract, (video, frames_dir, video_dir))
     for video in video_list]

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
