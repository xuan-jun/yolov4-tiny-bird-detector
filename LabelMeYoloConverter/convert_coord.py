import os
from os import walk, getcwd
from PIL import Image
from classes import *


""" Configure Paths"""   
mypath = "./test/"

wd = getcwd()

for label in os.listdir(mypath):
    if label.endswith("]"):
        current_path = os.path.join(mypath, label+"/")
        # checking for json files
        json_name_list = []
        for file in os.listdir(current_path):
            if file.endswith(".json"):
                json_name_list.append(file)

        # changing each json file

        for json_name in json_name_list:
            txt_name = json_name.rstrip(".json") + ".txt"
            """ Open input text files """
            txt_path = current_path + json_name
            print("Input:" + txt_path)
            txt_file = open(txt_path, "r")
            
            """ Open output text files """
            txt_outpath = current_path + txt_name
            print("Output:" + txt_outpath)
            txt_outfile = open(txt_outpath, "a")

            """ Convert the data to YOLO format """ 
            lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
            for idx, line in enumerate(lines):
                if ("lineColor" in line):
                    break 	#skip reading after find lineColor
                if ("label" in line):
                    print(line[16:-2])
                    x1 = float(lines[idx+3].rstrip(','))
                    y1 = float(lines[idx+4])
                    x2 = float(lines[idx+7].rstrip(','))
                    y2 = float(lines[idx+8])

                #in case when labelling, points are not in the right order
                    xmin = min(x1,x2)
                    xmax = max(x1,x2)
                    ymin = min(y1,y2)
                    ymax = max(y1,y2)
                    b = (xmin, xmax, ymin, ymax)
                    
                    txt_outfile.write(" ".join([str(a) for a in b]) + '\n')
        
            txt_file.close()
            os.remove(os.path.join(current_path, json_name))