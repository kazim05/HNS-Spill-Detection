#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import shutil
import argparse
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from core.config import cfg

parser = argparse.ArgumentParser(description="Run HNS spill detection on a folder of images using a frozen .pb graph")
parser.add_argument("--image_dir", type=str, required=True,
                    help="folder containing the images to run inference on")
parser.add_argument("--pb_file", type=str, default="./yolov3_hns.pb",
                    help="frozen graph produced by freeze_graph.py")
parser.add_argument("--result_dir", type=str, default="./result",
                    help="folder for the annotated output images (wiped on each run)")
parser.add_argument("--input_size", type=int, default=576,
                    help="network input resolution")
parser.add_argument("--score_threshold", type=float, default=cfg.TEST.SCORE_THRESHOLD,
                    help="minimum class score to keep a box")
parser.add_argument("--iou_threshold", type=float, default=cfg.TEST.IOU_THRESHOLD,
                    help="NMS IoU threshold")
flag = parser.parse_args()

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
num_classes     = len(utils.read_class_names(cfg.YOLO.CLASSES))
image_paths     = sorted(os.listdir(flag.image_dir))
graph           = tf.Graph()

if os.path.exists(flag.result_dir): shutil.rmtree(flag.result_dir)
os.makedirs(flag.result_dir)

# Load the frozen graph once and reuse the session for every image
return_tensors = utils.read_pb_return_tensors(graph, flag.pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    for image_path in image_paths:
        original_image = cv2.imread(os.path.join(flag.image_dir, image_path))
        if original_image is None:
            print("=> skipping (not a readable image): %s" % image_path)
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [flag.input_size, flag.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, flag.input_size, flag.score_threshold)
        bboxes = utils.nms(bboxes, flag.iou_threshold, method='nms')
        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        # image.show()
        image.save(os.path.join(flag.result_dir, os.path.basename(image_path)))
        print("=> %s: %d detection(s)" % (image_path, len(bboxes)))

print("=> Results written to: %s" % flag.result_dir)
