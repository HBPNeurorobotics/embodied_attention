#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import sys
import rospy
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import StereoCameraModel
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import Point
from tf2_geometry_msgs import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf2_ros
import traceback
from skimage.draw import circle

class Curiosity():
    def __init__(self):
        saliency_map_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)
        camera_info_left_sub = rospy.Subscriber("/hollie/camera/left/camera_info", CameraInfo, self.camera_info_left_callback, queue_size=1, buff_size=2**24)
        camera_info_right_sub = rospy.Subscriber("/hollie/camera/right/camera_info", CameraInfo, self.camera_info_right_callback, queue_size=1, buff_size=2**24)
        point_sub = rospy.Subscriber("/saccade_point", PointStamped, self.saccade_point_callback, queue_size=1, buff_size=2**24)
        disparity_sub = rospy.Subscriber("/hollie/camera/disparity", DisparityImage, self.disparity_callback, queue_size=1, buff_size=2**24)

        self.saliency_map_curiosity_pub = rospy.Publisher("/saliency_map_curiosity", Float32MultiArray, queue_size=1)
        self.saliency_map_curiosity_image_pub = rospy.Publisher("/saliency_map_curiosity_image", Image, queue_size=1)

        self.camera_info_left = None
        self.camera_info_right = None
        self.disparity_image = None
        self.camera_model = StereoCameraModel()

        self.targets = []

        self.cv_bridge = CvBridge()

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))
        self.min_disparity = rospy.get_param("/hollie/camera/stereo_image_proc/min_disparity", "-16")

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(30))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def saliency_map_callback(self, saliency_map):
        if self.camera_info_left is not None and self.camera_info_right is not None and self.disparity_image is not None:
            # handle input
            lo = saliency_map.layout
            saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

            # modify saliency map
            self.camera_model.fromCameraInfo(self.camera_info_left, self.camera_info_right)
            disparity_image = self.cv_bridge.imgmsg_to_cv2(self.disparity_image.image)
            for target in self.targets:
                target.header.stamp = rospy.Time.now()
                try:
                    transformed = self.tfBuffer.transform(target, self.camera_model.tfFrame(), timeout=rospy.Duration(0.1))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.loginfo("Transformation gone wrong")
                    traceback.print_exc()
                    return
                point_torso = (-transformed.point.y, -transformed.point.z, transformed.point.x)
                pixel = self.camera_model.project3dToPixel(point_torso)
                x = int(pixel[0][0] * (self.saliency_width/float(self.camera_info_left.width)))
                y = int(pixel[0][1] * (self.saliency_height/float(self.camera_info_left.height)))
                disparity = self.camera_model.getDisparity(point_torso[2])
                x = x + disparity
                if x >= 0 and x < self.saliency_width and y >=0 and y < self.saliency_height:
                    rr, cc = circle(y, x, 15, (len(saliency_map), len(saliency_map[0])))
                    saliency_map[rr, cc] = 0.

            self.saliency_map_curiosity_pub.publish(Float32MultiArray(layout=lo, data=saliency_map.flatten()))

            saliency_map_image = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255.
            saliency_map_image = np.uint8(saliency_map_image)

            try:
                saliency_map_curiosity_image = self.cv_bridge.cv2_to_imgmsg(saliency_map_image, "mono8")
            except CvBridgeError as e:
                print e

            self.saliency_map_curiosity_image_pub.publish(saliency_map_curiosity_image)

        else:
            rospy.loginfo("Received saliency map but camera information is missing")

    def camera_info_left_callback(self, camera_info):
        self.camera_info_left = camera_info

    def camera_info_right_callback(self, camera_info):
        self.camera_info_right = camera_info

    def disparity_callback(self, disparity_image):
        self.disparity_image = disparity_image

    def saccade_point_callback(self, point):
        self.targets.append(point)

def main(args):
    rospy.init_node("curiosity")
    curiosity = Curiosity()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
