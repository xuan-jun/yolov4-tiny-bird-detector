import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
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
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        # models are loaded while we evaluate each image
        # weights of the 5 models
        models = [tf.saved_model.load("./checkpoints/yolov4-garden-416", tags=[tag_constants.SERVING]),
                   tf.saved_model.load("./checkpoints/yolov4-kingfisher-416",tags=[tag_constants.SERVING]),
                   tf.saved_model.load("./checkpoints/yolov4-raptor-416", tags=[tag_constants.SERVING]),
                   tf.saved_model.load("./checkpoints/yolov4-waterbird-416", tags=[tag_constants.SERVING]),
                   tf.saved_model.load("./checkpoints/yolov4-woodpecker-416", tags=[tag_constants.SERVING])]

        first_model = tf.saved_model.load("./checkpoints/yolov4-first-416", tags=[tag_constants.SERVING])

        boxes, scores, classes, valid_detections = [np.zeros(1), np.zeros(shape = (1, 2)), np.zeros(1), np.zeros(1)]
        batch_data = tf.constant(images_data)
        class_num = 0
        
        # run through first model
        #start timing
        tic = time.perf_counter()
        
        infer = first_model.signatures['serving_default']
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
        temp_boxes, temp_scores, temp_classes, temp_valid_detections = [temp_boxes.numpy(), temp_scores.numpy(), temp_classes.numpy(), temp_valid_detections.numpy()] 
        max_score = 0
        for i in range(temp_valid_detections[0]):
            if temp_scores[0][i] >= temp_scores[0][max_score]:
                max_score = i

        class_num = int(temp_classes[0][max_score])
        
        # when we have a detected class
        if temp_valid_detections[0] != 0:
            infer = models[class_num].signatures['serving_default']
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                temp_boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(temp_boxes, (tf.shape(temp_boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            
            # end timing
            t = time.perf_counter() - tic
            print(f'Time taken is {t} seconds')
                
        # if we do not have a detected class
        else:
            for i in range(5):

                infer = models[i].signatures['serving_default']
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
        
        if class_num == 0:
            class_names = utils.read_class_names(cfg_garden.YOLO.CLASSES)
        elif class_num == 1:
            class_names = utils.read_class_names(cfg_kingfisher.YOLO.CLASSES)
        elif class_num == 2:
            class_name = utils.read_class_names(cfg_raptor.YOLO.CLASSES)
        elif class_num == 3:
            class_name = utils.read_class_names(cfg_waterbird.YOLO.CLASSES)
        else:
            class_name = utils.read_class_names(cfg_woodpecker.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

                
        if class_num == 0:
            image = utils_garden.draw_bbox(np.copy(original_image), [boxes, scores, classes, valid_detections])
        elif class_num == 1:
            image = utils_kingfisher.draw_bbox(np.copy(original_image), [boxes, scores, classes, valid_detections])
        elif class_num == 2:
            image = utils_raptor.draw_bbox(np.copy(original_image), [boxes, scores, classes, valid_detections])
        elif class_num == 3:
            image = utils_waterbird.draw_bbox(np.copy(original_image), [boxes, scores, classes, valid_detections])
        else:
            image = utils_woodpecker.draw_bbox(np.copy(original_image), [boxes, scores, classes, valid_detections])

        image = Image.fromarray(image.astype(np.uint8))
        if not FLAGS.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
