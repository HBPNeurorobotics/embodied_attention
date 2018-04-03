#!/usr/bin/env python
PKG = 'embodied_attention'

import roslib; roslib.load_manifest(PKG)
import rospy

import sys
import os
import numpy as np

from sensor_msgs.msg import Image
from embodied_attention.srv import Roi

from cv_bridge import CvBridge, CvBridgeError

tensorflow_path = rospy.get_param("tensorflow_path", os.path.expanduser('~') + "/.opt/tensorflow_venv/lib/python2.7/site-packages")
model_path = rospy.get_param("~model_path", os.path.expanduser('~') + '/.opt/models/research')
graph_path = rospy.get_param("~graph_path", os.path.expanduser('~') + '/.opt/graph_def/frozen_inference_graph.pb')

import site
site.addsitedir(tensorflow_path)
import tensorflow as tf
site.addsitedir(model_path)
from object_detection.utils import label_map_util

class Recognize():
  def __init__(self):
    self.cv_bridge = CvBridge()

    # paths to saved model states, update these paths if different in your local installation
    sys.path.append(model_path)
    sys.path.append(model_path + '/object_detection')
    sys.path.append(model_path + '/slim')

    label_path = model_path + '/object_detection/data/mscoco_label_map.pbtxt'

    # initialize the detection graph
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)

    # create internal label and category mappings
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=90,
                                                                use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

    s = rospy.Service('recognize', Roi, self.recognize)

  def recognize(self, roi):
    try:
      frame = self.cv_bridge.imgmsg_to_cv2(roi.RequestRoi, "bgr8")
    except CvBridgeError as e:
      print e

    numpy_image = np.expand_dims(frame, axis=0)

    # run the actual detection
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: numpy_image})

    boxes, scores, classes, num_detections = map(
      np.squeeze, [boxes, scores, classes, num_detections])

    for i in range(0, 5):
        print self.category_index[classes[i]]['name'] + ": " + str(scores[i])
    print

    return self.category_index[classes[0]]['name']

def main(args):
  rospy.init_node("recognize")
  Recognize()
  rospy.spin()

if __name__ == '__main__':
  main(sys.argv)
