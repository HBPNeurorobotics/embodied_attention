#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import sys
import rospy
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
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
        saccade_target_sub = rospy.Subscriber("/saccade_target", Point, self.saccade_target_callback, queue_size=1, buff_size=2**24)
        camera_info_sub = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1, buff_size=2**24)

        self.saliency_map_curiosity_pub = rospy.Publisher("/saliency_map_curiosity", Float32MultiArray, queue_size=1)
        self.saliency_map_curiosity_image_pub = rospy.Publisher("/saliency_map_curiosity_image", Image, queue_size=1)

        self.camera_info = None
        self.targets = []

        self.cv_bridge = CvBridge()
        self.camera_model = PinholeCameraModel()

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(30))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def saliency_map_callback(self, saliency_map):
        if self.camera_info is not None:
            # handle input
            lo = saliency_map.layout
            saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

            # modify saliency map
            self.camera_model.fromCameraInfo(self.camera_info)
            for target in self.targets:
                t = PointStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "hollie_arm_base_mimic_upper_torso_link"
                t.point.x = target[0]
                t.point.y = target[1]
                t.point.z = target[2]
                try:
                    transformed = self.tfBuffer.transform(t, 'camera_left_link_optical', timeout=rospy.Duration(0.1))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.loginfo("Transformation gone wrong")
                    traceback.print_exc()
                    return
                point_torso = (transformed.point.x, transformed.point.y, transformed.point.z)
                pixel = self.camera_model.project3dToPixel(point_torso)
                x = int(pixel[0] * (self.saliency_width/float(self.camera_info.width)))
                y = int(pixel[1] * (self.saliency_height/float(self.camera_info.height)))
                if x >= 0 and x < self.saliency_width and y >=0 and y < self.saliency_height:
                    rr, cc = circle(y, x, 10)
                    saliency_map[rr, cc] = -1.

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

    def camera_info_callback(self, camera_info):
        self.camera_info = camera_info

    def saccade_target_callback(self, saccade):
        if self.camera_info is not None:
            # scale to camera image size
            x = int(saccade.x * (float(self.camera_info.width)/self.saliency_width))
            y = int(saccade.y * (float(self.camera_info.height)/self.saliency_height))
    
            # compute target in polar coordinates
            self.camera_model.fromCameraInfo(self.camera_info)
            point_eye = self.camera_model.projectPixelTo3dRay((x, y))
    
            # transform to static torso frame
            t = PointStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "camera_left_link_optical"
            t.point.x = point_eye[0]
            t.point.y = point_eye[1]
            t.point.z = 10.0
            try:
                transformed = self.tfBuffer.transform(t, 'hollie_arm_base_mimic_upper_torso_link', timeout=rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.loginfo("Transformation gone wrong")
                traceback.print_exc()
                return
            point_torso = (transformed.point.x, transformed.point.y, transformed.point.z)
            self.targets.append(point_torso)
        else:
            rospy.loginfo("Received saliency map but camera information is missing")

def main(args):
    rospy.init_node("curiosity")
    curiosity = Curiosity()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
