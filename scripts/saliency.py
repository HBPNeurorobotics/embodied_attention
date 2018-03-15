#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from attention import Saliency
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os

class SaliencyNode():
    def __init__(self):
        tensorflow_path = rospy.get_param("~tensorflow_path", os.path.expanduser('~') + "/.opt/tensorflow_venv/lib/python2.7/site-packages")
        model_file = rospy.get_param('~saliency_file', '/tmp/model.ckpt')
        network_input_height = float(rospy.get_param('~network_input_height', '192'))
        network_input_width = float(rospy.get_param('~network_input_width', '256'))

        self.saliency = Saliency(tensorflow_path, model_file, network_input_height, network_input_width)

        image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.saliency_map_pub = rospy.Publisher("/saliency_map", Float32MultiArray, queue_size=1)
        self.saliency_map_image_pub = rospy.Publisher("/saliency_map_image", Image, queue_size=1)

        self.cv_bridge = CvBridge()

    def __del__(self):
        self.sess.close()

    def image_callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print e

        saliency_map = self.saliency.compute_saliency_map(image)

        height = MultiArrayDimension(size=len(saliency_map))
        width = MultiArrayDimension(size=len(saliency_map[0]))
        lo = MultiArrayLayout([height, width], 0)

        self.saliency_map_pub.publish(Float32MultiArray(layout=lo, data=saliency_map.flatten()))

        try:
            saliency_map_image = self.cv_bridge.cv2_to_imgmsg(np.uint8(saliency_map * 255.), "mono8")
        except CvBridgeError as e:
            print e

        self.saliency_map_image_pub.publish(saliency_map_image)

def main():
    rospy.init_node("saliency")
    saliency_node = SaliencyNode()
    rospy.spin()

if __name__ == "__main__":
    main()
