import os
import numpy as np
import cv2
import torchvision.transforms as T
from skimage.feature import hog

def load(filepath):
    data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        print("Oooops, cant parse %s, skipping this one..." , filepath)

    return data

pathvideo = "C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/Motsynth/MOTSynth_1/000.mp4"
filepath = "C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/MOT/mot_annotations/000/gt/gt.txt"
output_dir = "C:/ReIDDataset_PR/"
data = load(filepath)
video = cv2.VideoCapture(pathvideo)

if not os.path.isdir(output_dir) or len(os.listdir(output_dir)) != 1800:
    os.makedirs(output_dir, exist_ok=True)

textfile = open(output_dir + "/dataset.txt", "w")

max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

f = 1

while f < max_frame - 10:
    print("Frame " + str(f))
    video.set(1, f)
    ret, frame = video.read()
    if not ret:
        print("Impossible read video")
        break
    else:
        pass

    thisF = np.flatnonzero(data[:,0]==f)
    for i in thisF:
        left = ((data[i, 2]-1)).astype(int)
        top = ((data[i, 3]-1)).astype(int)
        width = ((data[i, 4])).astype(int)
        heigth = ((data[i, 5])).astype(int)

        crop_img = frame[top:top+heigth,left:left+width]

        if(crop_img.shape[0] > 60 and crop_img.shape[1] > 60):
            #cv2.imshow("First", crop_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            print("Person " + str(data[i, 1]))
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            target_res = (256, 128)
            pixels_per_cell = 8
            nbins = 9

            transform = T.Compose([
                    T.ToTensor(),
                    T.Resize(target_res)
            ])
            
            scaled = transform(crop_img).permute(1,2,0).numpy()
            fd, fd_vis = hog(scaled, orientations=nbins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                    cells_per_block=(1,1), visualize=True)
            

            next_f = f + 10
            video.set(1, next_f)
            ret, next_frame = video.read()
            if not ret:
                print("impossible read next frame")
                break
            else:
                pass

            
            nextF = np.flatnonzero(data[:,0]==next_f)
            print("Next frame" + str(next_f))
            for j in nextF:
                left = ((data[j, 2]-1)).astype(int)
                top = ((data[j, 3]-1)).astype(int)
                width = ((data[j, 4])).astype(int)
                heigth = ((data[j, 5])).astype(int)

                crop_img = next_frame[top:top+heigth,left:left+width]
                if(crop_img.shape[0] > 60 and crop_img.shape[1] > 60):
                    #cv2.imshow("Second", crop_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    scaled = transform(crop_img).permute(1,2,0).numpy()
                    fd_next, fd_vis_next = hog(scaled, orientations=nbins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                        cells_per_block=(1,1), visualize=True)
                    
                    np.savetxt(textfile, fd, newline= " ")
                    textfile.write("\n")
                    np.savetxt(textfile, fd_next, newline=" ")
                    textfile.write("\n")
                    if data[i, 1] == data[j, 1]:
                        textfile.write("1\n")
                    else:
                        textfile.write("0\n")

    f = f + 10




        
    
