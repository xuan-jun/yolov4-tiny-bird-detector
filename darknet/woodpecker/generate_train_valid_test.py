import os
from sklearn.model_selection import train_test_split

image_files = []
image_labels = []

os.chdir(os.path.join("data", "obj"))

# finding the image_files paths to prepare for train_test_split
for label in os.listdir(os.getcwd()):
    os.chdir(os.path.join(str(label)))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append(f"data/obj/{label}/" + filename)
            image_labels.append(label)
    os.chdir("..")
os.chdir("..")
 
# creating training, validation and testing sets
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1


# splits first time, 0.9 training+validation, 0.1 testing set
image_temp, image_test, label_temp, label_test = train_test_split(
    image_files, image_labels, test_size = test_ratio, random_state = 6, stratify=image_labels
)

# splits second time, 0.1 validation 0.8 training
image_train, image_valid, label_train, label_valid = train_test_split(
    image_temp, label_temp, test_size = validation_ratio/(validation_ratio+train_ratio), random_state = 12, stratify=label_temp
)

# writing the output files
with open("train.txt", "w") as outfile:
    for image in image_train:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
    
with open("valid.txt", "w") as outfile:
    for image in image_valid:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
    
with open("test.txt", "w") as outfile:
    for image in image_test:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()    
    
os.chdir("..")