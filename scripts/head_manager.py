#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from geometry_msgs.msg import PointStamped, Point
import geometry_msgs
from std_msgs.msg import Float64, String
import std_msgs
from std_srvs.srv import SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import PinholeCameraModel
from embodied_attention.srv import Roi, Look, Target
from ros_holographic.srv import NewObject, ProbeLabel, ProbeCoordinate
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import tf2_ros
import math
import numpy as np
import tf

from gazebo_msgs.msg import LinkStates

import time

class HeadManager():
    def __init__(self):
        self.tilt_eye_pub = rospy.Publisher("/eye_tilt", Float64, queue_size=1)
        self.pan_eye_left_pub = rospy.Publisher("/left_eye_pan", Float64, queue_size=1)
        self.pan_eye_right_pub = rospy.Publisher("/right_eye_pan", Float64, queue_size=1)
        self.tilt_head_pub = rospy.Publisher("/neck_pitch", Float64, queue_size=1)
        self.pan_head_pub = rospy.Publisher("/neck_yaw", Float64, queue_size=1)
        self.roi_pub = rospy.Publisher("/roi", Image, queue_size=1)
        self.label_pub = rospy.Publisher("/label", String, queue_size=1)
        self.probe_pub = rospy.Publisher("/probe_results", String, queue_size=1)
        self.point_pub = rospy.Publisher("/saccade_point", PointStamped, queue_size=1)
        self.tilt_pub = rospy.Publisher("/tilt", Float64, queue_size=1)
        self.pan_pub = rospy.Publisher("/pan", Float64, queue_size=1)
        self.shift_pub = rospy.Publisher("/shift", std_msgs.msg.Empty, queue_size=1)
        self.status_pub = rospy.Publisher("/status", String, queue_size=1)

        self.recognizer = rospy.ServiceProxy('recognize', Roi)
        self.memory = rospy.ServiceProxy('new_object', NewObject)
        self.probe_label = rospy.ServiceProxy('probe_label', ProbeLabel)
        self.probe_coordinate = rospy.ServiceProxy('probe_coordinate', ProbeCoordinate)

        self.saccade_ser = rospy.Service('saccade', Target, self.saccade)
        self.look_ser = rospy.Service('look', Look, self.look)
        self.probe_ser = rospy.Service('probe', SetBool, self.probe)

        self.camera_image = None
        self.camera_info_left = None
        self.camera_model = PinholeCameraModel()

        self.tilt_eye = 0.
        self.pan_eye = 0.
        self.tilt_head = 0.
        self.pan_head = 0.

        self.tilt_eye_upper_limit = 0.4869469
        self.tilt_eye_lower_limit = -0.8220501
        self.pan_eye_limit = 0.77754418
        self.tilt_head_limit = math.pi/4
        self.pan_head_limit = math.pi/2

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))
        self.move_eyes = rospy.get_param("~move_eyes", True)
        self.move_head = rospy.get_param("~move_head", True)
        self.shift = rospy.get_param("~shift", True)
        self.recognize = rospy.get_param("~recognize", True)
        self.probe = rospy.get_param("~probe", False)

        self.cv_bridge = CvBridge()

        camera_sub = rospy.Subscriber("/camera_left/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        camera_info_left_sub = rospy.Subscriber("/camera_left/camera_info", CameraInfo, self.camera_info_left_callback, queue_size=1, buff_size=2**24)
        joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1, buff_size=2**24)
        saccade_sub = rospy.Subscriber('saccade_target', Point, self.saccade, queue_size=3)

    def saccade(self, saccade):
        if self.camera_image is None:
            rospy.loginfo("Received saccade but camera_image is missing")
            return False
        elif self.camera_info_left is None:
            rospy.loginfo("Received saccade but camera_info_left is missing")
            return False
        else:

            if isinstance(saccade, Target):
                # called with the service
                saccade = saccade.target

            # scale saccade target to camera image size
            x = int(saccade.x * (float(self.camera_image.width)/self.saliency_width))
            y = int(saccade.y * (float(self.camera_image.height)/self.saliency_height))

            # create and publish roi
            size = 25
            x1 = max(0, x - size)
            y1 = max(0, y - size)
            x2 = min(x + size, self.camera_image.width)
            y2 = min(y + size, self.camera_image.height)

            try:
                image = self.cv_bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            except CvBridgeError as e:
                print e
            roi = image[y1:y2, x1:x2]
            try:
                roi = self.cv_bridge.cv2_to_imgmsg(roi, "bgr8")
            except CvBridgeError as e:
                print e
            self.roi_pub.publish(roi)

            # get point in eye frame
            self.camera_model.fromCameraInfo(self.camera_info_left)
            point = self.camera_model.projectPixelTo3dRay((x, y))

            point_s = PointStamped()
            point_s.header.stamp = rospy.Time.now()
            point_s.point.x = point[0] + self.pan_head + self.pan_eye
            point_s.point.y = point[1] + self.tilt_head + self.tilt_eye
            point_s.point.z = point[2]
            self.point_pub.publish(point_s)

            point = (point[2], point[0], -point[1])

            print "self.tilt_eye: %f" % self.tilt_eye
            print "self.pan_eye: %f" % self.pan_eye
            print "self.tilt_head: %f" % self.tilt_head
            print "self.pan_head: %f" % self.pan_head

            pan = math.atan2(point[1], point[0])
            tilt = math.atan2(-point[2], math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2)))

            pan_all = self.pan_eye + self.pan_head + pan
            tilt_all = self.tilt_eye + self.tilt_head + tilt

            self.pan_pub.publish(pan_all)
            self.tilt_pub.publish(tilt_all)

            print "tilt: %f" % tilt
            print "pan: %f" % pan

            pan_eye = self.pan_eye + pan
            tilt_eye = self.tilt_eye + tilt

            pan_head = self.pan_head + pan
            tilt_head = self.tilt_head + tilt

            print "tilt_eye: %f" % tilt_eye
            print "pan_eye: %f" % pan_eye
            print "tilt_head: %f" % tilt_head
            print "pan_head: %f" % pan_head

            if self.move_eyes:
                if abs(pan_eye) < self.pan_eye_limit and self.tilt_eye_lower_limit < tilt_eye and tilt_eye < self.tilt_eye_upper_limit:
                    rospy.loginfo("moving eyes only")
                    self.pan_eye_left_pub.publish(pan_eye)
                    self.pan_eye_right_pub.publish(pan_eye)
                    self.tilt_eye_pub.publish(tilt_eye)
                elif self.move_head:
                    # trim head values
                    tilt_head_trimmed = tilt_head
                    pan_head_trimmed = pan_head
                    if tilt_head < -self.tilt_head_limit:
                        rospy.loginfo("Tilt_head under limit, trimming")
                        tilt_head_trimmed = -self.tilt_head_limit
                    elif tilt_head > self.tilt_head_limit:
                        rospy.loginfo("Tilt_head over limit, trimming")
                        tilt_head_trimmed = self.tilt_head_limit
                    if pan_head < -self.pan_head_limit:
                        rospy.loginfo("Pan_head under limit, trimming")
                        pan_head_trimmed = -self.pan_head_limit
                    elif pan_head > self.pan_head_limit:
                        rospy.loginfo("Pan_head over limit, trimming")
                        pan_head_trimmed = self.pan_head_limit

                    print "tilt_head_trimmed: %f" % tilt_head_trimmed
                    print "pan_head_trimmed: %f" % pan_head_trimmed

                    # move head
                    rospy.loginfo("moving head")
                    self.pan_head_pub.publish(-pan_head_trimmed)
                    self.tilt_head_pub.publish(tilt_head_trimmed)

                    pan_eye = pan_eye - (pan_head_trimmed - self.pan_head)
                    tilt_eye = tilt_eye - (tilt_head_trimmed - self.tilt_head)
                    print "new tilt_eye: %f" % tilt_eye
                    print "new pan_eye: %f" % pan_eye

                    if abs(pan_eye) < self.pan_eye_limit and self.tilt_eye_lower_limit < tilt_eye and tilt_eye < self.tilt_eye_upper_limit:
                        rospy.loginfo("moving eyes")
                        self.pan_eye_left_pub.publish(pan_eye)
                        self.pan_eye_right_pub.publish(pan_eye)
                        self.tilt_eye_pub.publish(tilt_eye)
                    else:
                        rospy.loginfo("Over eye joint limit even though we moved head, dropping")
                        self.status_pub.publish("dropping")
                        return False
                else:
                    rospy.loginfo("Over eye joint limit and not moving head, dropping")
                    self.status_pub.publish("dropping")
                    return False
            else:
                rospy.loginfo("Not moving eyes, dropping")
                self.status_pub.publish("dropping")
                return False

            print

            if self.shift:
                # shift activity
                self.shift_pub.publish(std_msgs.msg.Empty())

            if not self.recognize:
                return True

            try:
                # object recognition
                ans = self.recognizer(roi)
                label = ans.Label
                certainty = ans.Certainty
                self.label_pub.publish(label)
                rospy.loginfo("Got label %s with certainty %f" % (label, certainty))

                mem_x = pan_all / (2 * math.pi) * 100
                mem_y = tilt_all / (2 * math.pi) * 100

                if self.probe:
                    probe_ans = self.probe_coordinate(mem_x, mem_y)
                    if probe_ans.return_value and len(probe_ans.Label) > 0:
                        if probe_ans.Label[0] == label:
                            res = "Approved: %s still at the same place" % label
                        else:
                            res = "Changed: Found %s where %s was before" % (label, probe_ans.Label[0])
                    else:
                        res = "New: Found %s, which wasn't here before" % label
                    res = res + " at %d, %d" % (mem_x, mem_y)
                    rospy.loginfo(res)
                    self.probe_pub.publish(res)
                # store in memory
                self.memory(mem_x, mem_y, label)
                rospy.loginfo("Stored in memory at %d, %d" % (mem_x, mem_y))
            except rospy.ServiceException:
                rospy.loginfo("Recognize or memory service call failed")

        print
        return True

    def image_callback(self, camera_image):
        self.camera_image = camera_image

    def camera_info_left_callback(self, camera_info_left):
        self.camera_info_left = camera_info_left

    def camera_info_right_callback(self, camera_info_right):
        self.camera_info_right = camera_info_right

    def joint_state_callback(self, joint_state):
        self.pan_eye = joint_state.position[joint_state.name.index("left_eye_pan")]
        self.tilt_eye = joint_state.position[joint_state.name.index("eye_tilt")]
        self.pan_head = -joint_state.position[joint_state.name.index("neck_yaw")]
        self.tilt_head = joint_state.position[joint_state.name.index("neck_pitch")]

    # TODO: rework, adapt to global pan/tilt values
    def look(self, label):
        # ask memory for label
        p = self.probe_label(label.Label)

        if not p.return_value or len(p.X) == 0 or len(p.Y) == 0:
            return False

        x = p.X[0]
        y = p.Y[0]

        pan = x / 100 * (2 * math.pi)
        tilt = y / 100 * (2 * math.pi)

        # adjust camera
        if abs(pan) < self.pan_eye_limit and self.tilt_eye_lower_limit < tilt and tilt < self.tilt_eye_upper_limit:
            self.pan_eye_left_pub.publish(pan)
            self.pan_eye_right_pub.publish(pan)
            self.tilt_eye_pub.publish(tilt)
            return True
        else:
            rospy.loginfo("Cannot look at " + label.Label + ", over eye joint limit and don't know how to move head yet..")
            return False

    def probe(self, value):
        self.probe = value.data
        return (True, 'success')

def main(args):
    rospy.init_node("head_manager")
    head_manager = HeadManager()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
