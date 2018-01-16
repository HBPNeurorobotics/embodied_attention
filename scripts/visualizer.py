#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import sys

class Visualizer():
    def __init__(self):
        image_sub = rospy.Subscriber("/icub_model/left_eye_camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        target_sub = rospy.Subscriber("/saccade_target", Point, self.target_callback, queue_size=1, buff_size=2**24)
        potential_target_sub = rospy.Subscriber("/saccade_potential_target", Point, self.potential_target_callback, queue_size=1, buff_size=2**24)

        pan_sub = rospy.Subscriber("/robot/left_eye_pan/pos", Float64, self.pan_callback, queue_size=1, buff_size=2**24)
        tilt_sub = rospy.Subscriber("/robot/eye_tilt/pos", Float64, self.tilt_callback, queue_size=1, buff_size=2**24)

        self.image_augmented_pub = rospy.Publisher("/image_visualized", Image, queue_size=1)

        self.image = None

        self.target = None
        self.potential_target = None

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

        self.cv_bridge = CvBridge()

    def target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(self.image.width)/self.saliency_width))
        y = int(target.y * (float(self.image.height)/self.saliency_height))

        self.target = (x, y)
        self.publish()

    def potential_target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(self.image.width)/self.saliency_width))
        y = int(target.y * (float(self.image.height)/self.saliency_height))

        self.potential_target = (x, y)
        self.publish()

    def publish(self):
        if self.image is not None:
            try:
                image = self.cv_bridge.imgmsg_to_cv2(self.image, "bgr8")
            except CvBridgeError as e:
                print e

            # visualization
            if self.target is not None:
                cv.circle(image, (self.target[0], self.target[1]), 2, (0, 0, 255))
            if self.potential_target is not None:
                cv.circle(image, (self.potential_target[0], self.potential_target[1]), 2, (0, 255, 0))

            try:
                image = self.cv_bridge.cv2_to_imgmsg(image, "bgr8")
            except CvBridgeError as e:
                print e

            self.image_augmented_pub.publish(image)
        else:
            print "Visualizer: received target but no image"

    def image_callback(self, image):
        self.image = image

    def pan_callback(self, pos):
        self.target = None
        self.potential_target = None

    def tilt_callback(self, pos):
        self.target = None
        self.potential_target = None

def main(args):
    rospy.init_node("visualizer")
    visualizer = Visualizer()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
