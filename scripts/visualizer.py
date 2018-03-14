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

class Visualizer():
    def __init__(self):
        self.image_visualized_pub = rospy.Publisher("/image_visualized", Image, queue_size=1)

        self.image = None

        self.target = None
        self.potential_target = None

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

        self.cv_bridge = CvBridge()

        image_sub = rospy.Subscriber("/hollie/camera/left/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        saliency_map_image_sub = rospy.Subscriber("/saliency_map_image", Image, self.saliency_map_image_callback, queue_size=1, buff_size=2**24)

        target_sub = rospy.Subscriber("/saccade_target", Point, self.target_callback, queue_size=1, buff_size=2**24)
        potential_target_sub = rospy.Subscriber("/saccade_potential_target", Point, self.potential_target_callback, queue_size=1, buff_size=2**24)

        pan_sub = rospy.Subscriber("/robot/hollie_left_eye_pan_joint/cmd_pos", Float64, self.pan_callback, queue_size=1, buff_size=2**24)
        tilt_sub = rospy.Subscriber("/robot/hollie_eyes_tilt_joint/cmd_pos", Float64, self.tilt_callback, queue_size=1, buff_size=2**24)

    def target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(len(self.image[0]))/self.saliency_width))
        y = int(target.y * (float(len(self.image))/self.saliency_height))
        self.target = (x, y)
        self.potential_target = None

    def potential_target_callback(self, target):
        # scale to camera image size
        x = int(target.x * (float(len(self.image[0]))/self.saliency_width))
        y = int(target.y * (float(len(self.image))/self.saliency_height))
        self.potential_target = (x, y)
        self.target = None

    def image_callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print e

        self.image = image.copy()

    def saliency_map_image_callback(self, saliency_map_image):
        if self.image is None:
            return

        try:
            saliency_map_image = self.cv_bridge.imgmsg_to_cv2(saliency_map_image, "mono8")
        except CvBridgeError as e:
            print e

        saliency_map_image = rescale_image(saliency_map_image, len(self.image), len(self.image[0]))
        saliency_map_image = saliency_map_image/255.
        saliency_map_image_3 = np.float64(np.zeros_like(self.image))
        saliency_map_image_3[:,:,0] = saliency_map_image
        saliency_map_image_3[:,:,1] = saliency_map_image
        saliency_map_image_3[:,:,2] = saliency_map_image

        combined = np.uint8(np.float64(self.image) * saliency_map_image_3)

        # visualization
        if self.target is not None:
            cv.circle(combined, (self.target[0], self.target[1]), 2, (0, 0, 255))
        if self.potential_target is not None:
            cv.circle(combined, (self.potential_target[0], self.potential_target[1]), 2, (0, 255, 0))

        try:
            combined = self.cv_bridge.cv2_to_imgmsg(combined, "bgr8")
        except CvBridgeError as e:
            print e

        self.image_visualized_pub.publish(combined)

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
