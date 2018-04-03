#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import sys
import rospy
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
from image_geometry import StereoCameraModel
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf
import tf2_ros
from stereo_msgs.msg import DisparityImage
from tf2_geometry_msgs import PointStamped
import math
from skimage.draw import circle

from gazebo_msgs.msg import LinkStates

class Modifier():
    def __init__(self):
        self.saliency_map_modified_pub = rospy.Publisher("/saliency_map_modified", Float32MultiArray, queue_size=1)
        self.saliency_map_modified_image_pub = rospy.Publisher("/saliency_map_modified_image", Image, queue_size=1)

        self.camera_image = None
        self.camera_info_left = None
        self.camera_info_right = None
        self.disparity_image = None
        self.camera_model = StereoCameraModel()
        # self.camera_model = PinholeCameraModel()
        self.link_states = None

        self.tilt_eye = 0.
        self.pan_eye = 0.
        self.tilt_head = 0.
        self.pan_head = 0.

        self.tilt_eye_limit = 0.925025
        self.pan_eye_limit = 0.925025
        self.tilt_head_limit = 1.57
        self.pan_head_limit = 1.57

        self.cv_bridge = CvBridge()

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(30))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.tilt_head_frame = 'hollie_neck_pitch_link'
        self.pan_head_frame = 'hollie_neck_yaw_link'
        self.tilt_eye_frame = 'swing_mesh'
        self.pan_eye_frame = 'camera_left_link'

        self.move_head = rospy.get_param("~move_head", False)
        self.move_eyes = rospy.get_param("~move_eyes", True)
        self.min_disparity = rospy.get_param("/hollie/camera/stereo_image_proc/min_disparity", "-16")

        saliency_map_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)
        camera_info_left_sub = rospy.Subscriber("/hollie/camera/left/camera_info", CameraInfo, self.camera_info_left_callback, queue_size=1, buff_size=2**24)
        camera_info_right_sub = rospy.Subscriber("/hollie/camera/right/camera_info", CameraInfo, self.camera_info_right_callback, queue_size=1, buff_size=2**24)
        disparity_sub = rospy.Subscriber("/hollie/camera/disparity", DisparityImage, self.disparity_callback, queue_size=1, buff_size=2**24)
        link_state_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_state_callback, queue_size=1, buff_size=2**24)
        joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1, buff_size=2**24)

    def saliency_map_callback(self, saliency_map):
        if self.camera_info_left is not None and self.camera_info_right is not None and self.disparity_image is not None:
            # handle input
            lo = saliency_map.layout
            saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

            # modify saliency map
            disparity_image = self.cv_bridge.imgmsg_to_cv2(self.disparity_image.image)
            self.camera_model.fromCameraInfo(self.camera_info_left, self.camera_info_right)
            for i in range(0, len(saliency_map)):
                for j in range(0, len(saliency_map[0])):
                    x = int(j * (float(self.camera_info_left.width)/len(saliency_map[0])))
                    y = int(i * (float(self.camera_info_left.height)/len(saliency_map)))

                    disparity = disparity_image[y][x]
                    if disparity < self.min_disparity:
                        disparity = 1.
                    point_eye = self.camera_model.projectPixelTo3d((x, y), disparity)

                    point_eye = (point_eye[2], point_eye[0], -point_eye[1])
                    pan_eye = self.pan_eye + math.atan2(point_eye[1], point_eye[0])
                    tilt_eye = self.tilt_eye + math.atan2(-point_eye[2], math.sqrt(math.pow(point_eye[0], 2) + math.pow(point_eye[1], 2)))

                    if self.move_head:
                        # TODO: tricky since we cannot move the head to find out
                        print "TODO"
                    else:
                        if abs(pan_eye) > self.pan_eye_limit or abs(tilt_eye) > self.tilt_eye_limit:
                            saliency_map[y][x] = 0.

            self.saliency_map_modified_pub.publish(Float32MultiArray(layout=lo, data=saliency_map.flatten()))

            try:
                saliency_map_modified_image = self.cv_bridge.cv2_to_imgmsg(np.uint8(saliency_map * 455.), "mono8")
            except CvBridgeError as e:
                print e

            self.saliency_map_modified_image_pub.publish(saliency_map_modified_image)
            print "done"

        else:
            rospy.loginfo("Received saliency map but camera information is missing")

    def camera_info_left_callback(self, camera_info):
        self.camera_info_left = camera_info

    def camera_info_right_callback(self, camera_info):
        self.camera_info_right = camera_info

    def disparity_callback(self, disparity_image):
        self.disparity_image = disparity_image

    def joint_state_callback(self, joint_state):
        self.pan_eye = joint_state.position[joint_state.name.index("hollie_left_eye_pan_joint")]
        self.tilt_eye = joint_state.position[joint_state.name.index("hollie_eyes_tilt_joint")]

    def link_state_callback(self, link_states):
        self.link_states = link_states

def main(args):
    rospy.init_node("modifier")
    modifier = Modifier()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
