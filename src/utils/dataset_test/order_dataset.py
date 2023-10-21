import shutil
import random
import os

def shuffle_dataset(filename):
    print("shuffling list")
    file = open(filename) 
    text_list = file.readlines()
    random.shuffle(text_list)
    return text_list

def copy_files(file_list, dest):
    for file in file_list:
        file_path, name= os.path.split(file)
        name = os.path.splitext(name)[0]
        shutil.copy2(file_path+ "/" + name + ".jpeg", dest + name + '.jpeg')
        shutil.copy2(file_path+ "/" + name + ".txt",  dest + name +'.txt')

def main():
    img_list = shuffle_dataset(r"D:\\trainhead.txt")
    copy_files(img_list[:7000],       r'D:\\hollywoods_heads/train/')
    copy_files(img_list[7000:9000],   r'D:\\hollywoods_heads/test/')
    copy_files(img_list[9000:10000],  r'D:\\hollywoods_heads/val/')

if __name__=="__main__": 
    main() 