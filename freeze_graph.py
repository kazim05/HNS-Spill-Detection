#! /usr/bin/env python
# coding=utf-8

import argparse
import tensorflow as tf
from core.yolov3 import YOLOV3
from core.config import cfg

parser = argparse.ArgumentParser(description="Freeze a YOLOv3 checkpoint into a .pb graph for image_demo.py")
parser.add_argument("--ckpt_file", type=str, default=cfg.TEST.WEIGHT_FILE,
                    help="checkpoint to freeze (default: cfg.TEST.WEIGHT_FILE)")
parser.add_argument("--pb_file", type=str, default="./yolov3_hns.pb",
                    help="output frozen graph (default: ./yolov3_hns.pb)")
flag = parser.parse_args()

# Must stay in sync with return_elements in image_demo.py
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)
print('=> Building the YOLOv3 graph from: %s' % flag.ckpt_file)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, flag.ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    input_graph_def=sess.graph.as_graph_def(),
    output_node_names=output_node_names)

with tf.gfile.GFile(flag.pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
print('=> Frozen graph written to: %s' % flag.pb_file)
