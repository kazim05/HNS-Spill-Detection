#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import shutil
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_hns.pb"
image_file_path = "/home/kazim/Desktop/hns/" 
image_paths     = os.listdir(image_file_path)
num_classes     = 3
input_size      = 576
graph           = tf.Graph()
result_path     = "./result/"

if os.path.exists(result_path): shutil.rmtree(result_path)
os.mkdir(result_path)

for image_path in image_paths:
    original_image = cv2.imread(os.path.join(image_file_path,image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


    with tf.Session(graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    # image.show()
    image.save(result_path + "/" + image_path.split("/")[-1])




