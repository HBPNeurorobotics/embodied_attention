#!/usr/bin/env python

import sys
import os
from os import path
import cv2 as cv
import numpy as np

def rescale_image(img, new_h, new_w):
    img_h, img_w = img.shape[:2]

    if img_h > new_h or img_w > new_w:
        interpolation = cv.INTER_AREA
    else:
        interpolation = cv.INTER_CUBIC

    aspect_ratio = img_w / img_h
    target_ratio = new_w / new_h
    height_ratio = new_h / img_h
    width_ratio = new_w / img_w

    if aspect_ratio != target_ratio:
        if width_ratio > height_ratio:
            new_w = np.floor(new_h * aspect_ratio)
        elif width_ratio < height_ratio:
            new_h = np.floor(new_w / aspect_ratio)

    rescaled_img = cv.resize(img, (int(new_w), int(new_h)),
                             interpolation=interpolation)
    return rescaled_img

def pad_image(img, new_h, new_w):
    img_h, img_w = img.shape[:2]

    if len(img.shape) == 3:
        pad_value = [126]*3
    else:
        pad_value = 0

    pad_vert = np.abs(new_h - img_h) / 2
    pad_horz = np.abs(new_w - img_w) / 2

    pad_t = np.floor(pad_vert).astype(int)
    pad_b = np.ceil(pad_vert).astype(int)
    pad_l = np.floor(pad_horz).astype(int)
    pad_r = np.ceil(pad_horz).astype(int)

    padded_img = cv.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                   borderType=cv.BORDER_CONSTANT,
                                   value=pad_value)
    return padded_img

class Saliency():
    def __init__(self, tensorflow_path=None,
                 use_gpu=False,
                 network_input_height=240, network_input_width=320, clip=False):
        if use_gpu:
            device = 'gpu'
        else:
            device = 'cpu'
        model_file=path.join(path.dirname(__file__), '..', '..', 'model', device, 'model.pb')

        if (tensorflow_path):
            import site
            site.addsitedir(tensorflow_path)
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        self.tf = tf

        ### saliency
        graph_def = tf.GraphDef()

        with tf.gfile.Open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        self.input = graph.get_operation_by_name("Placeholder").outputs[0]
        self.output = graph.get_operation_by_name("transpose_9").outputs[0]

        self.sess = tf.Session(graph=graph)

        self.network_input_height = int(network_input_height)
        self.network_input_width = int(network_input_width)
        self.clip = bool(clip)

    def __del__(self):
        self.sess.close()

    def compute_saliency_map(self, frame):
        ### compute saliency map
        stim = rescale_image(frame, self.network_input_height, self.network_input_width)
        stim = pad_image(stim, self.network_input_height, self.network_input_width)
        stim = stim[None, :, :, :].transpose(0, 3, 1, 2)

        # self.tf.reset_default_graph()

        saliency_map = self.sess.run(self.output, feed_dict={self.input: stim})
        saliency_map = saliency_map.squeeze()
        if self.clip:
            saliency_map = saliency_map.clip(0)
        else:
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        return saliency_map
