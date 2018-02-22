#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import sys
import rospy
from std_msgs.msg import Float64, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Modifier():
    def __init__(self):
        saliency_map_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)
        camera_info_sub = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.camera_info_callback, queue_size=1, buff_size=2**24)
        joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1, buff_size=2**24)

        self.saliency_map_modified_pub = rospy.Publisher("/saliency_map_modified", Float32MultiArray, queue_size=1)
        self.saliency_map_modified_image_pub = rospy.Publisher("/saliency_map_modified_image", Image, queue_size=1)

        self.eye_joint_limit = 0.942477796

        self.camera_info = None
        self.pan = 0.
        self.tilt = 0.

        self.cv_bridge = CvBridge()
        self.camera_model = PinholeCameraModel()

    def saliency_map_callback(self, saliency_map):
        if self.camera_info is not None:
            # handle input
            lo = saliency_map.layout
            saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

            # modify saliency map
            self.camera_model.fromCameraInfo(self.camera_info)
            for y in range(0, len(saliency_map)):
                for x in range(0, len(saliency_map[0])):
                    cam_x = x * (float(self.camera_info.width)/len(saliency_map[0]))
                    cam_y = y * (float(self.camera_info.height)/len(saliency_map))
                    ray = self.camera_model.projectPixelTo3dRay((cam_x, cam_y))
                    dx = float(np.arctan(-ray[0]))
                    dy = float(np.arctan(-ray[1] / np.sqrt(ray[0] * ray[0] + 1)))
                    if self.pan + dx < -self.eye_joint_limit or self.pan + dx > self.eye_joint_limit or self.tilt + dy < -self.eye_joint_limit or self.tilt + dy > self.eye_joint_limit:
                        saliency_map[y][x] = -1.

            self.saliency_map_modified_pub.publish(Float32MultiArray(layout=lo, data=saliency_map.flatten()))

            saliency_map_image = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min()) * 255.
            saliency_map_image = np.uint8(saliency_map_image)

            try:
                saliency_map_modified_image = self.cv_bridge.cv2_to_imgmsg(saliency_map_image, "mono8")
            except CvBridgeError as e:
                print e

            self.saliency_map_modified_image_pub.publish(saliency_map_modified_image)

        else:
            rospy.loginfo("Received saliency map but camera information is missing")

    def camera_info_callback(self, camera_info):
        self.camera_info = camera_info

    def joint_state_callback(self, joint_state):
        self.pan = joint_state.position[joint_state.name.index("left_eye_pan")]
        self.tilt = joint_state.position[joint_state.name.index("eye_tilt")]

def main(args):
    rospy.init_node("modifier")
    modifier = Modifier()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
