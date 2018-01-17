#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
import cv2 as cv
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
import os

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
    def __init__(self):
        model_file = rospy.get_param('~saliency_file', '/tmp/model.ckpt')

        meta_file = model_file + ".meta"
        index_file = model_file + ".index"
        data_file = model_file + ".data-00000-of-00001"

        if (not os.path.exists(meta_file) or not os.path.exists(index_file) or not os.path.exists(data_file)):
            rospy.logwarn("Model files not present:\n\t{}\n\t{}\n\t{}\nWe will download them from owncloud."
                .format(meta_file, index_file, data_file))
            import wget
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download("https://neurorobotics-files.net/owncloud/index.php/s/TNpWFSX8xLvfbYD/download", meta_file)
            wget.download("https://neurorobotics-files.net/owncloud/index.php/s/sDCFUGTrzJyhDA5/download", index_file)
            wget.download("https://neurorobotics-files.net/owncloud/index.php/s/Scti429S7D11tMv/download", data_file)

        self.saliency_pub = rospy.Publisher("/saliency_map", Float32MultiArray, queue_size=1)
        self.saliency_image_pub = rospy.Publisher("/saliency_map_image", Image, queue_size=1)
        self.cv_bridge = CvBridge()

        self.sess = tf.Session()
        self.net = tf.train.import_meta_graph(meta_file)
        self.net.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self.output = graph.get_operation_by_name("conv2d_8/BiasAdd").outputs[0]
        self.input = graph.get_tensor_by_name("Placeholder_1:0")

        image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        # network has been trained and works best on 192x256
        # both height and width must be divisible by 8
        self.network_input_height = float(rospy.get_param('~network_input_height', '192'))
        self.network_input_width = float(rospy.get_param('~network_input_width', '256'))

    def __del__(self):
        self.sess.close()

    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print e

        stim = rescale_image(frame, self.network_input_height, self.network_input_width)
        stim = pad_image(stim, self.network_input_height, self.network_input_width)
    
        stim = stim[None, :, :, :]
        stim = stim.astype(np.float32)
    
        # subtract mean channel activation
        stim[:, :, :, 0] -= 99.50135
        stim[:, :, :, 1] -= 109.85075
        stim[:, :, :, 2] -= 118.14625

        stim = stim.transpose(0, 3, 1, 2)

        tf.reset_default_graph()

        saliency_map = self.sess.run(self.output, feed_dict={self.input: stim})

        saliency_map = saliency_map.squeeze()

        # publish saliency map
        dim0 = MultiArrayDimension(size=len(saliency_map))
        dim1 = MultiArrayDimension(size=len(saliency_map[0]))
        data_offset = 0

        lo = MultiArrayLayout([dim0, dim1], data_offset)

        self.saliency_pub.publish(Float32MultiArray(layout=lo, data=saliency_map.flatten()))

        # publish saliency map image
        saliency_map_image = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255.
        saliency_map_image = np.uint8(saliency_map_image)

        try:
            res_msg = self.cv_bridge.cv2_to_imgmsg(saliency_map_image, "mono8")
        except CvBridgeError as e:
            print e

        self.saliency_image_pub.publish(res_msg)


def main():
    rospy.init_node("saliency")
    saliency = Saliency()
    rospy.spin()


if __name__ == "__main__":
    main()
