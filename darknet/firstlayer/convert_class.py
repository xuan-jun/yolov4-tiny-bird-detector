import os
import numpy as np

data_path = "data/obj"

garden = [0, 1, 2, 4, 5, 6, 8, 9, 14, 16, 18, 24, 27, 28, 29, 30, 31, 32, 34, 36, 37, 46, 47]
kingfisher = [12, 15, 39, 44]
raptors = [7, 10, 20, 23, 38, 42]
waterbirds = [3, 11, 17, 21, 22, 26, 33, 40, 43, 45]
woodpeckers = [13, 19, 25, 35, 41]

vals = [[str(i) for i in garden], [str(i) for i in kingfisher], [str(i) for i in raptors], [str(i) for i in waterbirds], [str(i) for i in woodpeckers]]

# loop over each class
for label in os.listdir(data_path):
    current_path = os.path.join(data_path, label+"/")
    # file names
    file_names = []    
    for file in os.listdir(current_path):
        if file.endswith(".txt"):
            # finding the file names
            file_names.append(file)
            
    # reading in the file
    for file_name in file_names:
        with open(current_path+file_name, 'r') as file :
            filedata = file.readlines()

        # Write the file out again
        with open(current_path+file_name, 'w') as file:
            for line in filedata:
                line = line.split(" ")
                # checking which group it belongs to
                for i in range(len(vals)):
                    if line[0] in vals[i]:
                        line[0] = str(i)
                        break
                file.write(" ".join(line))