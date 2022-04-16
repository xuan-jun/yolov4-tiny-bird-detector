# YOLOv4-tiny Bird Detecotr

# Model

**YOLOv4-tiny**

*  **References**:
    
    * Darknet Framework: https://github.com/AlexeyAB/darknet

        * Implementation [here](https://github.com/xuan-jun/yolov4-tiny-bird-detector/tree/main/darknet)
    
    * Tensorflow Framework: https://github.com/theAIGuysCode/tensorflow-yolov4-tflite

        * Implementation [here](https://github.com/xuan-jun/yolov4-tiny-bird-detector/tree/main/tensorflow)

# Dataset

## Species

There is a total of 48 Bird Species that we are aiming to detect and they can be broken up into the following classes:

1. Garden and Urban Birds - 23 Species

2. Kingfishers - 4 Species

3. Raptors - 6 Species

4. Waterbirds - 10 Species

5. Woodpeckers - 5 Species

| No. | Bird Species | Class |
| ----------- | ----------- | ----------- |
|1| Ashy Tailorbird | Garden and Urban Birds |
|2| Asian Glossy Starling | Garden and Urban Birds |
|3| Asian Koel | Garden and Urban Birds |
|4| Black-crowned night heron | Waterbirds |
|5| Black-naped Oriole | Garden and Urban Birds |
|6| Blue-tailed Bee-eater | Garden and Urban Birds |
|7| Blue-throated Bee-eater | Garden and Urban Birds |
|8| Brahminy kite | Raptors |
|9| Brown Shrike | Garden and Urban Birds |
|10| Brown-throated Sunbird | Garden and Urban Birds |
|11| Buffy fish owl | Raptors |
|12| Chinese pond heron | Waterbirds |
|13| Collared Kingfisher | Kingfishers |
|14| Common Flameback | Woodpeckers |
|15| Common Iora | Garden and Urban Birds |
|16| Common Kingfisher | Kingfishers |
|17| Common Myna | Garden and Urban Birds |
|18| Common sandpiper | Waterbirds |
|19| Common Tailorbird | Garden and Urban Birds |
|20| Coppersmith Barbet | Woodpeckers |
|21| Crested serpent eagle | Raptors |
|22| Eastern cattle egret | Waterbirds |
|23| Grey heron | Waterbirds |
|24| Grey-headed fish eagle | Raptors |
|25| Javan Myna | Garden and Urban Birds |
|26| Laced Woodpecker | Woodpeckers |
|27| Little egret | Waterbirds |
|28| Malaysian Pied Fantail | Garden and Urban Birds |
|29| Olive-backed Sunbird | Garden and Urban Birds |
|30| Oriental Magpie-Robin | Garden and Urban Birds |
|31| Oriental Pied Hornbill | Garden and Urban Birds |
|32| Pacific Swallow | Garden and Urban Birds |
|33| Pink-necked Green Pigeon | Garden and Urban Birds |
|34| Purple heron | Waterbirds |
|35| Rose-ringed Parakeet | Garden and Urban Birds |
|36| Rufous Woodpecker | Woodpeckers |
|37| Savanna Nightjar | Garden and Urban Birds |
|38| Scarlet-backed Flowerpecker | Garden and Urban Birds |
|39| Spotted wood owl | Raptors |
|40| Stork-billed Kingfisher | Kingfishers |
|41|Striated heron | Waterbirds |
|42|Sunda Pygmy Woodpecker | Woodpeckers |
|43|White-bellied sea eagle | Raptors |
|44|White-breasted waterhen | Waterbirds |
|45|White-throated Kingfisher | Kingfishers |
|46|Yellow bittern | Waterbirds |
|47|Yellow-vented Bulbul | Garden and Urban Birds |
|48|Zebra Dove | Garden and Urban Birds |

## Labeling of Images
Images were labelled using [labelme](https://github.com/wkentaro/labelme) 

## Conversion of Images

### YOLO Format

Converted using the convert.py under the LabelMeYoloConverter folder (referenced from [here](https://github.com/ivder/LabelMeYoloConverter)).

**Format**

The corresponding bounding box for the ground truth were labelled with <!-- $(x_{centerbb}, y_{centerbb}, w_{widthbb}, h_{heightbb})$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=(x_%7Bcenterbb%7D%2C%20y_%7Bcenterbb%7D%2C%20w_%7Bwidthbb%7D%2C%20h_%7Bheightbb%7D)">

Where:

* <!-- $x_{centerbb}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_%7Bcenterbb%7D"> - x-coordinate of the center of the bounding box
* <!-- $y_{centerbb}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y_%7Bcenterbb%7D"> - y-coordinate of the center of the bounding box
* <!-- $w_{widthbb}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=w_%7Bwidthbb%7D"> - width of the bounding box
* <!-- $h_{heightbb}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=h_%7Bheightbb%7D"> - height of the bounding box


# Detection and Evaluation

## YOLOv4-tiny

Note that the YOLOv4-tiny model requires a dedicated GPU for the local machine, if there isn't any, Google Collab can also be used for running the code.

### Darknet Framework

**Code for running on Google Collab**

File can be found [here](https://github.com/xuan-jun/yolov4-tiny-bird-detector/tree/main/darknet/yolov4.ipynb). Referenced from [here](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)

**Code for running on Local Machine:**

Instructions for setting up darknet on local machine can be found [here](https://github.com/AlexeyAB/darknet) and various instructions for preparing dataset and config files.

```

# intial training
darknet.exe detector train data/obj-tiny.data cfg/yolov4-tiny-obj.cfg yolov4-tiny.conv.29 -map

# rerun with the last saved weights
darknet.exe detector train data/obj-tiny.data cfg/yolov4-tiny-obj.cfg backup/yolov4-tiny-obj_last.weights -map

# check mAP score
darknet.exe detector map data/obj-tiny.data cfg/yolov4-tiny-obj.cfg backup/yolov4-obj_best.weights -map

# running detection on custom object
darknet.exe detector test data/obj-tiny.data cfg/yolov4-tiny-obj.cfg backup/yolov4-tiny-obj_best.weights data/obj/ -thresh 0.3

```

### Tensorflow Framework

Training is carried out under the darknet framework and can be converted to the Tensorflow model where there are 3 implementations:

1. Base Model

2. Ensemble Implementation

3. Filtered Implementation

```

# converting

## converting darknet weights into tensorflow
python save_model.py --weights ./data/yolov4-best.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

# running yolo tensorflow model

## base model
python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --images ./data/images/bird.jpg --tiny

## ensemble
python detect_ensemble.py --size 416 --model yolov4 --images ./data/images/bird.jpg --tiny

## filtered
python detect_filtered.py --size 416 --model yolov4 --images ./data/images/bird.jpg --tiny

# evaluation

## base model
python evaluate.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --annotation_path ./data/test.txt --tiny

## ensemble
python evaluate_ensemble.py --size 416 --model yolov4 -- tiny

## filtered
python evaluate_filtered.py --size 416 --model yolov4 -- tiny

```