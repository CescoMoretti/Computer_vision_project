import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import torchvision.transforms as T

from skimage.feature import hog
from skimage import data, exposure

img_path = os.path.join("C:\people01", "People_4.0.jpg")
img_2 = os.path.join("C:\people01" ,"People_12.0.jpg")

original_img = cv2.imread(img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
person_2 = cv2.imread(img_2)
person_2 = cv2.cvtColor(person_2, cv2.COLOR_BGR2RGB)

target_res = (256,128)
pixel_per_cells = 8
rows = int(target_res[0]/pixel_per_cells)
cols = int(target_res[1]/pixel_per_cells)
nbins = 9

transfor = T.Compose([
    T.ToTensor(),
    T.Resize(target_res)
    ])


scaled = transfor(original_img).permute(1,2,0).numpy()
scaled_2 = transfor(person_2).permute(1,2,0).numpy()

fd, fd_vis = hog(scaled, orientations=nbins, pixels_per_cell=(pixel_per_cells, pixel_per_cells),
                    cells_per_block=(1,1), visualize=True)
fd_2, fd_vis_2 = hog(scaled_2, orientations=nbins, pixels_per_cell=(pixel_per_cells, pixel_per_cells),
                    cells_per_block=(1,1), visualize=True)

hog_image_rescaled = exposure.rescale_intensity(fd_vis, in_range=(0, 10))
hog_image_rescaled_2 = exposure.rescale_intensity(fd_vis_2, in_range=(0, 10))

print("Feature vector Hog", fd)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 7))
ax[0][0].imshow(original_img)
ax[0][1].imshow(scaled)
ax[0][2].hist(fd)
ax[0][3].imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax[1][0].imshow(person_2)
ax[1][1].imshow(scaled_2)
ax[1][2].hist(fd_2)
ax[1][3].imshow(hog_image_rescaled_2, cmap=plt.cm.gray)
plt.show()