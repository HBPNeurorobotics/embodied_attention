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

        self.targets = []
        self.potential_targets = []

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

    def target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(self.image.width)/self.saliency_width))
        y = int(target.y * (float(self.image.height)/self.saliency_height))

        self.targets.append((x, y))
        self.publish()

    def potential_target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(self.image.width)/self.saliency_width))
        y = int(target.y * (float(self.image.height)/self.saliency_height))

        self.potential_targets.append((x, y))
        self.publish()

    def publish(self):
        if self.image is not None:
            image = CvBridge().imgmsg_to_cv2(self.image, "bgr8")

            # visualization
            for t in self.potential_targets:
                cv.circle(image, (t[0], t[1]), 2, (0, 255, 0))
            for t in self.targets:
                cv.circle(image, (t[0], t[1]), 2, (0, 0, 255))

            image = CvBridge().cv2_to_imgmsg(image, "bgr8")
            self.image_augmented_pub.publish(image)
        else:
            print "but information is missing"

    def image_callback(self, image):
        self.image = image

    def pan_callback(self, pos):
        self.targets = []
        self.potential_targets = []

    def tilt_callback(self, pos):
        self.targets = []
        self.potential_targets = []

def main(args):
    rospy.init_node("visualizer")
    visualizer = Visualizer()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
