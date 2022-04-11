import os
import numpy as np
import cv2 as cv
from PIL import Image

data_path = "data/obj"
train_path = "train/"
valid_path = "valid/"
test_path = "test/"

for label in os.listdir(data_path):
    os.makedirs(os.path.join("train", label))
    os.makedirs(os.path.join("valid", label))
    os.makedirs(os.path.join("test", label))

# helper function to convert
def convert(x,y,w,h, dim_w, dim_h):
    x1, y1 = (x-w/2) * dim_w, (y-h/2) * dim_h
    x2, y2 = (x+w/2) * dim_w, (y+h/2) * dim_h
    return (x1, y1, x2, y2)

with open("data/train.txt", "r") as f1, open("data/valid.txt", "r") as f2, open("data/test.txt", "r") as f3:
    train_data = f1.readlines()
    valid_data = f2.readlines()
    test_data = f3.readlines()

# converting training data
for file in train_data:
    # removing newline character
    file = file.rstrip("\n")
    # extracting the file name
    file_name = file.split('/')[-1]
    # species
    species = file.split('/')[2]    
    
    # writing the img file to the training folder
    cv.imwrite(train_path+species+'/'+file_name, cv.imread(file))
    
    # getting dimensions of image
    with Image.open(file) as im:
       width, height = im.size
    
    # reading original .txt file
    txt_path = file.rstrip(".jpg")+".txt"
    with open(txt_path) as txt:
        bboxes = txt.readlines()
    
    txt_file_name = file_name.rstrip(".jpg")+".txt"
    with open(train_path+species+'/'+txt_file_name, "a") as new_txt:
        for bbox in bboxes:
            # stripping the newline character & split on space
            label, x, y, w, h = bbox.rstrip("\n").split(" ")
            # converting to x1, y1, x2, y2
            bb = convert(float(x), float(y), float(w), float(h), width, height)
            
            new_txt.write(" ".join([str(a) for a in bb]) + '\n')

# converting validation data
for file in valid_data:
    # removing newline character
    file = file.rstrip("\n")
    # extracting the file name
    file_name = file.split('/')[-1]
    # species
    species = file.split('/')[2]    
    
    # writing the img file to the valid folder
    cv.imwrite(valid_path+species+'/'+file_name, cv.imread(file))
    
    # getting dimensions of image
    with Image.open(file) as im:
       width, height = im.size
    
    # reading original .txt file
    txt_path = file.rstrip(".jpg")+".txt"
    with open(txt_path) as txt:
        bboxes = txt.readlines()
    
    txt_file_name = file_name.rstrip(".jpg")+".txt"
    with open(valid_path+species+'/'+txt_file_name, "a") as new_txt:
        for bbox in bboxes:
            # stripping the newline character & split on space
            label, x, y, w, h = bbox.rstrip("\n").split(" ")
            # converting to x1, y1, x2, y2
            bb = convert(float(x), float(y), float(w), float(h), width, height)
            
            new_txt.write(" ".join([str(a) for a in bb]) + '\n')

# converting test data
for file in test_data:
    # removing newline character
    file = file.rstrip("\n")
    # extracting the file name
    file_name = file.split('/')[-1]
    # species
    species = file.split('/')[2]    
    
    # writing the img file to the test folder
    cv.imwrite(test_path+species+'/'+file_name, cv.imread(file))
    
    # getting dimensions of image
    with Image.open(file) as im:
       width, height = im.size
    
    # reading original .txt file
    txt_path = file.rstrip(".jpg")+".txt"
    with open(txt_path) as txt:
        bboxes = txt.readlines()
    
    txt_file_name = file_name.rstrip(".jpg")+".txt"
    with open(test_path+species+'/'+txt_file_name, "a") as new_txt:
        for bbox in bboxes:
            # stripping the newline character & split on space
            label, x, y, w, h = bbox.rstrip("\n").split(" ")
            # converting to x1, y1, x2, y2
            bb = convert(float(x), float(y), float(w), float(h), width, height)
            
            new_txt.write(" ".join([str(a) for a in bb]) + '\n')