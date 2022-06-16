import cv2 as cv

capture = cv.VideoCapture('Videos/000.avi')
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