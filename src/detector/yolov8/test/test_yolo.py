from ultralytics import YOLO
from PIL import Image
from cap_from_youtube import cap_from_youtube
import cv2


def test_video(model):
    youtube_url = 'https://www.youtube.com/watch?v=DRNivy7rkTg'
    capture = cap_from_youtube(youtube_url, '1440p60')
    # Show the results

    while capture.isOpened():
        # Read a frame from the video
        success, frame = capture.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)
        
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    capture.release()
    cv2.destroyAllWindows()
    
def test_img(model):

    # Define path to the image file
    source = "./src/detector/yolov8/test/mov_005_071723.jpeg"
    img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    
    results = model.predict(img, classes=[0])
    box = results[0].boxes.cpu().numpy().xywhn[0]
    print(box)
    height = img.shape[0]
    width = img.shape[1]
    print(img.shape)

    box[0]*= width
    box[2]*= width
    box[1]*= height
    box[3]*= height
    
    print(box)
    print(results[0].boxes.cpu().numpy().xywh[0])



def main():
    # model = YOLO('best.pt')
    model = YOLO('yolov8n.pt')

    test_img(model)
    # test_video(model)


if __name__ == "__main__":
    main()
