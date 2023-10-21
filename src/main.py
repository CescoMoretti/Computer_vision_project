from pipeline_object.pipeline import Pipeline
import os
import argparse

this_path = os.getcwd

def main():
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument(
        '--video_path', help='path to the video', required=True)

    args = parser.parse_args()

    video_path = args.video_path
    print(video_path)
    pipeline = Pipeline(video_path)
    for i in range(500):
        pipeline.process_frame()

if __name__ == "__main__":
    main()