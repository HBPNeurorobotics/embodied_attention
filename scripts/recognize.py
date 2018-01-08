#!/usr/bin/env python
PKG = 'embodied_attention'

import roslib; roslib.load_manifest(PKG)
import rospy

import sys
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions

from embodied_attention.srv import Roi

import tensorflow as tf


height = 224
width = 224

def rescale(img, height, width):
  if len(img.shape) == 3:
    h, w, _ = img.shape
  elif len(img.shape) == 2:
    h, w = img.shape
  
  if (float(height)/h) > (float(width)/w):
    img = resize(img, (height, w*height/h),
                                     order=3, preserve_range=True)
  else:
    img = resize(img, (h*width/w, width),
                                     order=3, preserve_range=True)
  
  if len(img.shape) == 3:
    h, w, _ = img.shape
  elif len(img.shape) == 2:
    h, w = img.shape
  
  img = img[h//2-(height/2):h//2+(height/2), w//2-(width/2):w//2+(width/2)]
  
  if len(img.shape) == 3:
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
  
  return img

model = MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model._make_predict_function()
graph = tf.get_default_graph()

def recognize(roi):
  try:
    frame = CvBridge().imgmsg_to_cv2(roi.RequestRoi, "bgr8")
  except CvBridgeError, e:
    print e

  x = rescale(frame, height, width)
  x = x.astype(np.float32)

  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  with graph.as_default():
    preds = model.predict(x)
    preds = decode_predictions(preds, top=10)[0]
    print('Predicitions: ', str(preds))
    print('Identified as: ', preds[0][1])
    return preds[0][1]

def main(args):
  rospy.init_node("recognize")
  s = rospy.Service('recognize', Roi, recognize)
  rospy.spin()

if __name__ == '__main__':
  main(sys.argv)