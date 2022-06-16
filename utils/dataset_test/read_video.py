import cv2 as cv
#put the path in your pc
capture = cv.VideoCapture('c:/Users/cesco/OneDrive/Desktop/computer vision/Project group 25/MOTSynth_1/000.mp4') 
while(True):
    isTrue, frame = capture.read()
    if not isTrue:
        print("Can't receive frame. Exiting ...")
        break
    cv.imshow('video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()

cv.waitKey(0)