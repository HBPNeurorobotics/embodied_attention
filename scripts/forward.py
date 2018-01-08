#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# network has been trained and works best on 192x256
# both height and width must be divisible by 8
img_height = 192
img_width = 256

def rescale_image(img, tar_h, tar_w):
    img_h, img_w = img.shape[:2]

    if img_h > tar_h or img_w > tar_w:
        interpolation = cv.INTER_AREA
    else:
        interpolation = cv.INTER_CUBIC

    aspect_ratio = img_w / img_h
    target_ratio = tar_w / tar_h
    height_ratio = tar_h / img_h
    width_ratio = tar_w / img_w

    new_h, new_w = tar_h, tar_w

    if aspect_ratio != target_ratio:
        if width_ratio > height_ratio:
            new_w = np.floor(new_h * aspect_ratio)
        elif width_ratio < height_ratio:
            new_h = np.floor(new_w / aspect_ratio)

    rescaled_img = cv.resize(img, (int(new_w), int(new_h)),
                             interpolation=interpolation)
    return rescaled_img


def pad_image(img, tar_h, tar_w):
    img_h, img_w = img.shape[:2]

    if len(img.shape) == 3:
        pad_value = [126]*3
    else:
        pad_value = 0

    pad_vert = np.abs(tar_h - img_h) / 2
    pad_horz = np.abs(tar_w - img_w) / 2

    pad_t = np.floor(pad_vert).astype(int)
    pad_b = np.ceil(pad_vert).astype(int)
    pad_l = np.floor(pad_horz).astype(int)
    pad_r = np.ceil(pad_horz).astype(int)

    padded_img = cv.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                   borderType=cv.BORDER_CONSTANT,
                                   value=pad_value)
    return padded_img


class Saliency():
    def __init__(self):
        self.model_path = rospy.get_param('~saliency_tensorflow_file', '/tmp/model.ckpt')
        image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.saliency_pub = rospy.Publisher("/saliency_map", Image, queue_size=1)
        self.cv_bridge = CvBridge()


    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e

        stim = rescale_image(frame, img_height, img_width)
        stim = pad_image(stim, img_height, img_width)
    
        stim = stim[None, :, :, :]
        stim = stim.astype(np.float32)
    
        # subtract mean channel activation
        stim[:, :, :, 0] -= 99.50135
        stim[:, :, :, 1] -= 109.85075
        stim[:, :, :, 2] -= 118.14625

        stim = stim.transpose(0, 3, 1, 2)

        tf.reset_default_graph()

        with tf.Session() as sess:
            net = tf.train.import_meta_graph(self.model_path + ".meta")
            net.restore(sess, self.model_path)
    
            graph = tf.get_default_graph()
            output = graph.get_operation_by_name("conv2d_8/BiasAdd").outputs[0]
            input = graph.get_tensor_by_name("Placeholder_1:0")
            saliency = sess.run(output, feed_dict={input: stim})

        saliency = saliency.squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) * 255. # is this correct?
        saliency = np.uint8(saliency)

        res_msg = CvBridge().cv2_to_imgmsg(saliency, "mono8")
        self.saliency_pub.publish(res_msg)


def main():
    rospy.init_node("saliency")
    saliency = Saliency()
    rospy.spin()


if __name__ == "__main__":
    main()
