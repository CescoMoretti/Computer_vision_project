import sys
import os
import numpy as np
import cv2
import torchvision.transforms as T
from skimage.feature import hog
from skimage import exposure
from matplotlib import pyplot as plt
from scipy.spatial import distance

def load(filepath):
    data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        data = np.genfromtxt(filepath, delimiter=",")
    if data.ndim == 1:
        print("Oooops, cant parse %s, skipping this one..." , filepath)

    return data

class ReIDHOG():
	
    def __init__(self, video = None, output_dir = None, filepath=None, method=None):
        self.video = cv2.VideoCapture(video)
        self.output_dir = output_dir
        self.filepath = filepath
        self.data = load(self.filepath)
        self.method = method
        self.lasts = []

    def save_people(self, f=None):
        if not os.path.isdir(self.output_dir) or len(os.listdir(self.output_dir)) != 1800:
            os.makedirs(self.output_dir, exist_ok=True) 

        textfile = open(self.output_dir + "/metrics.txt", "a")

        if f == None:
            print("There isn't a frame count")
        if f > int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)):
            textfile.close()
            return False

        self.video.set(1, f)
        ret, frame = self.video.read()
        if not ret:
            print("Impossible read of the frame")
        else:
            pass

        text = "--------- FRAME " + str(f) + " ------------------\n"
        textfile.write(text)
        thisF = np.flatnonzero(self.data[:,0]==f)
        for i in thisF:
            left = ((self.data[i, 2]-1)).astype(int)
            top = ((self.data[i, 3]-1)).astype(int)
            width = ((self.data[i, 4])).astype(int)
            heigth = ((self.data[i, 5])).astype(int)

            crop_img = frame[top:top+heigth,left:left+width]

            if(crop_img.shape[0] > 60 and crop_img.shape[1] > 60):
                name = self.output_dir + "/" + str(self.data[i, 1])
                if not os.path.isdir(name):
                    print(name)
                    os.makedirs(name, exist_ok=True) 
                    name = name +  "/" + str(f) + ".jpg"
                    self.lasts.append(name)
                    cv2.imwrite(name, crop_img)
                else:
                    comparation = []
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
                    
                    hog_image_rescaled = exposure.rescale_intensity(fd_vis, in_range=(0, 10))

                    for file in self.lasts:
                        img = os.path.join(self.output_dir, file)
                        compare = cv2.imread(img)
                        compare = cv2.cvtColor(compare, cv2.COLOR_BGR2RGB)

                        scaled_compare = transform(compare).permute(1,2,0).numpy()
                        fd_compare, fd_vis_compare = hog(scaled_compare, orientations=nbins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(1,1), visualize=True)
                        
                        print("Shape of 1-D Vector (FD-NEW):")
                        print(fd_compare.shape)
                        
                        comparation.append(distance.euclidean(fd, fd_compare))
                        
                    print("Shape of 1-D Vector (FD):")
                    print(fd.shape)
                    comparation = np.array(comparation)
                    equal = comparation.argmin()
                    print("The Position is " + str(equal) + " The Element is " + self.lasts[equal])
                    element = self.lasts[equal].split("/", -1)
                    name = element[0] + "/" + element[1] + "/" + element[2] + "/" + element[3] + "/" + str(f) + ".jpg"
                    text = "The element " + str(self.data[i, 1]) + " is Equal to " + str(element[3]) + " With distance " + str(comparation[equal]) + "\n"
                    textfile.write(text)
                    self.lasts[equal] = name
                    cv2.imwrite(name, crop_img)

        textfile.close()
        return True
