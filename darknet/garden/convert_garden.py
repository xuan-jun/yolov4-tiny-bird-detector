import os
import numpy as np

data_path = "data/obj"

garden = [0, 1, 2, 4, 5, 6, 8, 9, 14, 16, 18, 24, 27, 28, 29, 30, 31, 32, 34, 36, 37, 46, 47]
vals = {}

for i in range(len(garden)):
    # assigning vals for the indexes of the bird
    vals[str(garden[i])] = str(i)

# loop over each class
for label in os.listdir(data_path):
    current_path = os.path.join(data_path, label+"/")
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
                # assigning the new index to the bird species
                line[0] = vals[line[0]]
                file.write(" ".join(line))