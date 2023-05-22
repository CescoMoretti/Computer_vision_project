from pipeline_object.pipeline import Pipeline
import os
import argparse

this_path = os.getcwd

def main():
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument(
        '--video_dir', help='Directory hosting videos', required=True)
    parser.add_argument('--out_dir', help='Directory of output', required=True)

    args = parser.parse_args()

    video_dir = args.video_dir
    out_dir = args.out_dir
    print(video_dir)
    pipeline = Pipeline(out_dir, video_dir)
    for i in range(500):
        pipeline.process_frame()

if __name__ == "__main__":
    main()