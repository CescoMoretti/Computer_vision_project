from ReIDHOG import ReIDHOG
import os

video = "C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/Motsynth/MOTSynth_1/000.mp4"
filepath = "C:/Users/gabri/OneDrive/Desktop/Universita/DaFare/Computer_Vision/Progetto/MOT/mot_annotations/000/gt/gt.txt"
method = "Euclidean"
output_dir = "C:/ReID/" + method

crop = ReIDHOG(video=video, output_dir=output_dir, filepath=filepath, method=method)

not_finish = True
f = 1

while not_finish:
    not_finish = crop.save_people(f)
    f = f + 5