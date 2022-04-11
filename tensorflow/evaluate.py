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
from core.config import cfg
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

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # # Build Model
    # if FLAGS.framework == 'tflite':
    #     interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    #     interpreter.allocate_tensors()
    #     input_details = interpreter.get_input_details()
    #     output_details = interpreter.get_output_details()
    #     print(input_details)
    #     print(output_details)
    # else:
    #     saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    #     infer = saved_model_loaded.signatures['serving_default']
    
    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            # annotation = line.strip().split()
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
            
            # bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])            
            
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
            image_size = image.shape[:2]
            # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            # loading the model
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            
            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
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
                    class_name = CLASSES[class_ind]
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


