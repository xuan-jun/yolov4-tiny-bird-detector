from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
import core.utils_garden as utils_garden
import core.utils_kingfisher as utils_kingfisher
import core.utils_raptor as utils_raptor
import core.utils_waterbird as utils_waterbird
import core.utils_woodpecker as utils_woodpecker
from core.config import cfg
from core.config_garden import cfg_garden
from core.config_kingfisher import cfg_kingfisher
from core.config_raptor import cfg_raptor
from core.config_waterbird import cfg_waterbird
from core.config_woodpecker import cfg_woodpecker
from PIL import Image

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/test.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

# helper function to convert
def convert(label, x,y,w,h, dim_w, dim_h):
    x1, y1 = (x-w/2) * dim_w, (y-h/2) * dim_h
    x2, y2 = (x+w/2) * dim_w, (y+h/2) * dim_h
    return (x1, y1, x2, y2, label)

def main(_argv):
    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    # because of the ensemble
    CLASSES_GARDEN = utils.read_class_names(cfg_garden.YOLO.CLASSES)
    CLASSES_KINGFISHER = utils.read_class_names(cfg_kingfisher.YOLO.CLASSES)
    CLASSES_RAPTOR = utils.read_class_names(cfg_raptor.YOLO.CLASSES)
    CLASSES_WATERBIRD = utils.read_class_names(cfg_waterbird.YOLO.CLASSES)
    CLASSES_WOODPECKER = utils.read_class_names(cfg_woodpecker.YOLO.CLASSES)

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    
    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            image_path = line.rstrip("\n")
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            
            with Image.open(image_path) as im:
                width, height = im.size
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with open(image_path.rstrip(".jpg")+".txt", 'r') as f:
              lines = f.readlines()
            temp = []
            for line in lines:
                label, x, y, w, h = line.rstrip("\n").split(" ")
                temp.append(convert(int(label), float(x), float(y), float(w), float(h), width, height))
            bbox_data_gt = np.array(temp)
   
            
            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]     
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            
            image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # loading the model
            # weights of the 5 models
            weights = ["./checkpoints/yolov4-garden-416",
                       "./checkpoints/yolov4-kingfisher-416",
                       "./checkpoints/yolov4-raptor-416",
                       "./checkpoints/yolov4-waterbird-416",
                       "./checkpoints/yolov4-woodpecker-416"]

            boxes, scores, classes, valid_detections = [np.zeros(1), np.zeros(shape = (1, 2)), np.zeros(1), np.zeros(1)]
            class_num = 0
            batch_data = tf.constant(image_data)
            for i in range(5):
            
                saved_model_loaded = tf.saved_model.load(weights[i], tags=[tag_constants.SERVING])
                infer = saved_model_loaded.signatures['serving_default']
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    temp_boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

                temp_boxes, temp_scores, temp_classes, temp_valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(temp_boxes, (tf.shape(temp_boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=FLAGS.iou,
                    score_threshold=FLAGS.score
                )

                if max(temp_scores.numpy()[0]) >= max(scores[0]):
                    class_num = i
                    boxes, scores, classes, valid_detections = [temp_boxes.numpy(), temp_scores.numpy(), temp_classes.numpy(), temp_valid_detections.numpy()]            
            
            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                if class_num == 0:
                    image_result = utils_garden.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                elif class_num == 1:
                    image_result = utils_kingfisher.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                elif class_num == 2:
                    image_result = utils_raptor.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                elif class_num == 3:
                    image_result = utils_waterbird.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                else:
                    image_result = utils_woodpecker.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])

                cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))


            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    # coor[0] = int(coor[0] * image_h)
                    # coor[2] = int(coor[2] * image_h)
                    # coor[1] = int(coor[1] * image_w)
                    # coor[3] = int(coor[3] * image_w)

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    if class_num == 0:
                        class_name = CLASSES_GARDEN[class_ind]
                    elif class_num == 1:
                        class_name = CLASSES_KINGFISHER[class_ind]
                    elif class_num == 2:
                        class_name = CLASSES_RAPTOR[class_ind]
                    elif class_num == 3:
                        class_name = CLASSES_WATERBIRD[class_ind]
                    else:
                        class_name = CLASSES_WOODPECKER[class_ind]

                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


