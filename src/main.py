from pipeline_object.pipeline import Pipeline
import os

this_path = os.getcwd()

def main():
    pipeline = Pipeline()
    for i in range(8):
        pipeline.process_frame()

if __name__ == "__main__":
    main()