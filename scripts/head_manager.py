#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, String
from std_srvs.srv import SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from embodied_attention.srv import Roi, Look
from ros_holographic.srv import NewObject, ProbeLabel, ProbeCoordinate
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import sys
import os
import numpy as np

class HeadManager():
    def __init__(self):
        saccade_sub = rospy.Subscriber("/saccade_target", Point, self.saccade_callback, queue_size=1, buff_size=2**24)
        camera_sub = rospy.Subscriber("/icub_model/left_eye_camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        camera_info_sub = rospy.Subscriber("/icub_model/left_eye_camera/camera_info", CameraInfo, self.camera_info_callback, queue_size=1, buff_size=2**24)

        self.pan_pub = rospy.Publisher("/robot/left_eye_pan/pos", Float64, queue_size=1)
        self.tilt_pub = rospy.Publisher("/robot/eye_tilt/pos", Float64, queue_size=1)
        self.roi_pub = rospy.Publisher("/roi", Image, queue_size=1)
        self.label_pub = rospy.Publisher("/label", String, queue_size=1)
        self.probe_pub = rospy.Publisher("/probe_results", String, queue_size=1)

        self.recognizer = rospy.ServiceProxy('recognize', Roi)
        self.memory = rospy.ServiceProxy('new_object', NewObject)
        self.probe_label = rospy.ServiceProxy('probe_label', ProbeLabel)
        self.probe_coordinate = rospy.ServiceProxy('probe_coordinate', ProbeCoordinate)

        self.look = rospy.Service('look', Look, self.look)
        self.mem = rospy.Service('memorize', SetBool, self.mem)

        self.camera_image = None
        self.camera_info = None
        self.camera_model = PinholeCameraModel()

        self.x = 0.
        self.y = 0.

        self.eye_joint_limit = 0.942477796

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

        self.cv_bridge = CvBridge()

        self.memorize = True

        self.counter = 0
        self.annotated_img_pub = rospy.Publisher('/annotated_image', Image, queue_size=1)
        self.image_saver = rospy.ServiceProxy('/hugin_panorama/image_saver/save', Empty)

    def saccade_callback(self, saccade):
        if self.camera_image is not None and self.camera_info is not None:
            # scale to camera image size
            x = int(saccade.x * (float(self.camera_image.width)/self.saliency_width))
            y = int(saccade.y * (float(self.camera_image.height)/self.saliency_height))

            # compute target in polar coordinates
            self.camera_model.fromCameraInfo(self.camera_info)
            ray = self.camera_model.projectPixelTo3dRay((x, y))
            dx = np.arctan(-ray[0])
            dy = np.arctan(-ray[1] / np.sqrt(ray[0] * ray[0] + 1))

            # publish new position
            if self.x + dx < -self.eye_joint_limit or self.x + dx > self.eye_joint_limit or self.y + dy < -self.eye_joint_limit or self.y + dy > self.eye_joint_limit:
                rospy.loginfo("\tOver eye joint limit, dropped")
                return

            self.x += dx
            self.y += dy
            self.pan_pub.publish(self.x)
            self.tilt_pub.publish(self.y)

            try:
                image = self.cv_bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            except CvBridgeError as e:
                print e

            annotated_img = image.copy()
            cv.putText(annotated_img, str(self.counter), (self.camera_info.width/2, self.camera_info.height/2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
            self.counter = self.counter + 1
            try:
                annotated_img = self.cv_bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            except CvBridgeError as e:
                print e
            self.annotated_img_pub.publish(annotated_img)
            self.image_saver()

            # set roi
            size = 25
            x1 = self.camera_info.width/2 - size
            y1 = self.camera_info.height/2 - size
            x2 = self.camera_info.width/2 + size
            y2 = self.camera_info.height/2 + size

            roi = image[y1:y2, x1:x2]

            try:
                roi = self.cv_bridge.cv2_to_imgmsg(roi, "bgr8")
            except CvBridgeError as e:
                print e

            self.roi_pub.publish(roi)

            try:
                # object recognition
                label = self.recognizer(roi).Label
                self.label_pub.publish(label)
                rospy.loginfo("\tGot label %s" % label)

                mem_x = self.x / self.eye_joint_limit * 100
                mem_y = self.y / self.eye_joint_limit * 100

                if self.memorize:
                    # store in memory
                    self.memory(mem_x, mem_y, label)
                    rospy.loginfo("\tStored in memory at %d, %d" % (mem_x, mem_y))
                else:
                    probe_ans = self.probe_coordinate(mem_x, mem_y)
                    if probe_ans.return_value and len(probe_ans.Label) > 0:
                        if probe_ans.Label[0] == label:
                            res = "Approved: %s still at the same place" % label
                        else:
                            res = "Changed: Found %s where %s was before" % (label, probe_ans.Label[0])
                    else:
                        res = "New: Found %s, which wasn't here before" % label
                    rospy.loginfo(res)
                    self.probe_pub.publish(res)
                    loc = "at %d, %d" % (mem_x, mem_y)
                    rospy.loginfo(loc)
                    self.probe_pub.publish(loc)
            except rospy.ServiceException:
                rospy.loginfo("\tRecognize or memory service call failed")

        else:
            rospy.loginfo("Received saccade but camera image or information is missing")

    def image_callback(self, camera_image):
        self.camera_image = camera_image

    def camera_info_callback(self, camera_info):
        self.camera_info = camera_info

    def look(self, label):
        # ask memory for label
        p = self.probe_label(label.Label)

        if not p.return_value or len(p.X) == 0 or len(p.Y) == 0:
            return False

        x = p.X[0]
        y = p.Y[0]

        # adjust camera
        self.pan_pub.publish(x/100.)
        self.tilt_pub.publish(y/100.)
        return True

    def mem(self, value):
        self.memorize = value.data
        return (True, 'success')

def main(args):
    rospy.init_node("head_manager")
    head_manager = HeadManager()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
