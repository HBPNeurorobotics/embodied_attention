#!/usr/bin/env python
PKG = 'embodied_attention'

import roslib; roslib.load_manifest(PKG)
import rospy

import os
import sys
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

import tensorflow as tf
from tensorflow.contrib import slim

from embodied_attention.srv import Roi

class Recognize():
    def __init__(self):
        self.cv_bridge = CvBridge()

        self.model_path = rospy.get_param('~model_path', '/tmp/')
        self.model_file = self.model_path + 'inception_v1.ckpt'

        if (not os.path.exists(self.model_file)):
            rospy.logwarn("Model files not present:\n\t{}\nWe will download them from tensorflow."
                .format(self.model_file))
            from datasets import dataset_utils
            url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
            dataset_utils.download_and_uncompress_tarball(url, self.model_path)

        self.names = imagenet.create_readable_names_for_imagenet_labels()

        s = rospy.Service('recognize', Roi, self.recognize)

    def recognize(self, roi):
      try:
        frame = self.cv_bridge.imgmsg_to_cv2(roi.RequestRoi, "bgr8")
      except CvBridgeError as e:
        print e

      image = tf.convert_to_tensor(frame)
      processed_image = inception_preprocessing.preprocess_image(image, inception.inception_v1.default_image_size, inception.inception_v1.default_image_size, is_training=False)
      processed_images = tf.expand_dims(processed_image, 0)

      with slim.arg_scope(inception.inception_v1_arg_scope()):
          logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
      probabilities = tf.nn.softmax(logits)

      init_fn = slim.assign_from_checkpoint_fn(
          self.model_file,
          slim.get_model_variables('InceptionV1'))

      with tf.Session() as sess:
          init_fn(sess)
          np_image, probabilities = sess.run([image, probabilities])
          probabilities = probabilities[0, 0:]
          sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

      for i in range(5):
          index = sorted_inds[i]
          rospy.loginfo('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, self.names[index]))

      return self.names[sorted_inds[0]].split(',')[0]

def main(args):
  rospy.init_node("recognize")
  Recognize()
  rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
