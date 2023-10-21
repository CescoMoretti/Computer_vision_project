from pipeline_object.pipeline import Pipeline
from functions.frame_extractor import Frame_extractor

import os
import argparse

this_path = os.getcwd

def main():
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument(
        '--video_path', help='path to the video', required=True)

    args = parser.parse_args()

    video_path = args.video_path    
    frame_extractor = Frame_extractor()
    frame_extractor.load_video(video_path)
    print("Loaded video from " + video_path)
    
    pipeline = Pipeline()

    video_ended = False
    while not video_ended:
        frame, video_ended = frame_extractor.extract()
        if video_ended:
            break
        pipeline.process_frame(frame)

if __name__ == "__main__":
    main()