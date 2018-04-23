#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from geometry_msgs.msg import Point
import geometry_msgs
from tf2_geometry_msgs import PointStamped
from std_msgs.msg import Float64, String
import std_msgs
from std_srvs.srv import SetBool, Empty
from sensor_msgs.msg import Image, CameraInfo, JointState
from image_geometry import StereoCameraModel
from stereo_msgs.msg import DisparityImage
from embodied_attention.srv import Roi, Look, Transform, Target
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
        self.tilt_eye_pub = rospy.Publisher("/hollie/eye_tilt_position_controller/command", Float64, queue_size=1)
        self.pan_eye_left_pub = rospy.Publisher("/hollie/left_eye_pan_position_controller/command", Float64, queue_size=1)
        self.pan_eye_right_pub = rospy.Publisher("/hollie/right_eye_pan_position_controller/command", Float64, queue_size=1)
        self.tilt_head_pub = rospy.Publisher("/hollie/neck_pitch_position_controller/command", Float64, queue_size=1)
        self.pan_head_pub = rospy.Publisher("/hollie/neck_yaw_position_controller/command", Float64, queue_size=1)
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

        self.saccade = rospy.Service('saccade', Target, self.saccade)
        self.look = rospy.Service('look', Look, self.look)
        self.mem = rospy.Service('memorize', SetBool, self.mem)
        self.transform = rospy.Service('transform', Transform, self.transform)

        self.camera_image = None
        self.camera_info_left = None
        self.camera_info_right = None
        self.disparity_image = None
        self.camera_model = StereoCameraModel()
        self.link_states = None

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
        self.min_disparity = rospy.get_param("/hollie/camera/stereo_image_proc/min_disparity", "-16")

        self.cv_bridge = CvBridge()

        self.memorize = True

        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(30))
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.transform = None
        self.static_frame = "hollie_base_x_link"

        camera_sub = rospy.Subscriber("/hollie/camera/left/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)
        camera_info_left_sub = rospy.Subscriber("/hollie/camera/left/camera_info", CameraInfo, self.camera_info_left_callback, queue_size=1, buff_size=2**24)
        camera_info_right_sub = rospy.Subscriber("/hollie/camera/right/camera_info", CameraInfo, self.camera_info_right_callback, queue_size=1, buff_size=2**24)
        disparity_sub = rospy.Subscriber("/hollie/camera/disparity", DisparityImage, self.disparity_callback, queue_size=1, buff_size=2**24)
        joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size=1, buff_size=2**24)
        link_state_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_state_callback, queue_size=1, buff_size=2**24)

    def saccade(self, saccade):
        if self.camera_image is None:
            rospy.loginfo("Received saccade but camera_image is missing")
            return False
        elif self.camera_info_left is None:
            rospy.loginfo("Received saccade but camera_info_left is missing")
            return False
        elif self.camera_info_right is None:
            rospy.loginfo("Received saccade but camera_info_right is missing")
            return False
        elif self.disparity_image is None:
            rospy.loginfo("Received saccade but disparity_image is missing")
            return False
        else:

            if self.transform is None:
                self.transform = self.tfBuffer.lookup_transform("camera_left_link_optical", self.static_frame, rospy.Time(0))

            saccade = saccade.target

            # scale saccade target to camera image size
            x = int(saccade.x * (float(self.camera_image.width)/self.saliency_width))
            y = int(saccade.y * (float(self.camera_image.height)/self.saliency_height))

            # get point in eye frame
            disparity_image = self.cv_bridge.imgmsg_to_cv2(self.disparity_image.image)
            disparity = disparity_image[y][x] - (self.min_disparity - 2)
            print "disparity: %f" % disparity
            self.camera_model.fromCameraInfo(self.camera_info_left, self.camera_info_right)
            distance = self.camera_model.getZ(disparity)
            point_eye = self.camera_model.projectPixelTo3d((x, y), disparity)

            point_eye = (point_eye[2], point_eye[0], -point_eye[1])
            print "point_eye: " + str(point_eye)

            pan_eye = self.pan_eye + math.atan2(point_eye[1], point_eye[0])
            tilt_eye = self.tilt_eye + math.atan2(-point_eye[2], math.sqrt(math.pow(point_eye[0], 2) + math.pow(point_eye[1], 2)))
            print "move_eye: %f, %f" % (tilt_eye, pan_eye)

            t = PointStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = self.camera_model.tfFrame()
            t.point.x = point_eye[0]
            t.point.y = -point_eye[1]
            t.point.z = point_eye[2]

            # publish as static point for curiosity
            point_static = self.tfBuffer.transform(t, self.static_frame, rospy.Duration(0.0001))
            self.point_pub.publish(point_static)

            # publish as point in original eye frame
            point_wide_angle = self.tfBuffer.registration.get(type(point_static))(point_static, self.transform)

            point_wide_angle = (point_wide_angle.point.x, point_wide_angle.point.y, point_wide_angle.point.z)
            pan = -math.atan2(point_wide_angle[1], point_wide_angle[0])
            tilt = math.atan2(-point_wide_angle[2], math.sqrt(math.pow(point_wide_angle[0], 2) + math.pow(point_wide_angle[1], 2)))
            self.pan_pub.publish(pan)
            self.tilt_pub.publish(tilt)

            # transform to head frame
            point_head = self.tfBuffer.transform(t, "base_link", rospy.Duration(0.0001))
            point_head = (point_head.point.x, point_head.point.y, point_head.point.z)
            print "point_head: " + str(point_head)

            pan_head = self.pan_head + math.atan2(point_head[1], point_head[0])
            tilt_head = self.tilt_head + math.atan2(-point_head[2], math.sqrt(math.pow(point_head[0], 2) + math.pow(point_head[1], 2)))
            print "move_head: %f, %f" % (tilt_head, pan_head)

            if self.move_eyes:
                if abs(pan_eye) < self.pan_eye_limit and self.tilt_eye_lower_limit < tilt_eye and tilt_eye < self.tilt_eye_upper_limit:
                    rospy.loginfo("Moving eyes: %f, %f" % (pan_eye, tilt_eye))
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

                    # move head
                    rospy.loginfo("Moving head: %f, %f" % (pan_head, tilt_head))
                    self.pan_head_pub.publish(pan_head_trimmed)
                    self.tilt_head_pub.publish(tilt_head_trimmed)

                    # transform to eye frame
                    point_eye = self.tfBuffer.transform(t, self.camera_model.tfFrame(), rospy.Duration(0.0001))
                    point_eye = (point_eye.point.x, point_eye.point.y, point_eye.point.z)
                    print "point_eye after head movement: " + str(point_eye)

                    pan_eye = self.pan_eye + math.atan2(point_eye[1], point_eye[0])
                    tilt_eye = self.tilt_eye + math.atan2(-point_eye[2], math.sqrt(math.pow(point_eye[0], 2) + math.pow(point_eye[1], 2)))
                    print "move_eye after head movement: %f, %f" % (tilt_eye, pan_eye)

                    if abs(pan_eye) < self.pan_eye_limit and self.tilt_eye_lower_limit < tilt_eye and tilt_eye < self.tilt_eye_upper_limit:
                        rospy.loginfo("Moving eyes: %f, %f" % (pan_eye, tilt_eye))
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

            if self.shift:
                # shift activity
                self.shift_pub.publish(std_msgs.msg.Empty())

            # create and publish roi
            size = 25
            x1 = x - size
            y1 = y - size
            x2 = x + size
            y2 = y + size

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

            try:
                # object recognition
                label = self.recognizer(roi).Label
                self.label_pub.publish(label)
                rospy.loginfo("Got label %s" % label)

                mem_x = pan / (2 * math.pi) * 100
                mem_y = tilt / (2 * math.pi) * 100

                if self.memorize:
                    # store in memory
                    self.memory(mem_x, mem_y, label)
                    rospy.loginfo("Stored in memory at %d, %d" % (mem_x, mem_y))
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
                rospy.loginfo("Recognize or memory service call failed")

        print
        return True

    def image_callback(self, camera_image):
        self.camera_image = camera_image

    def camera_info_left_callback(self, camera_info_left):
        self.camera_info_left = camera_info_left

    def camera_info_right_callback(self, camera_info_right):
        self.camera_info_right = camera_info_right

    def disparity_callback(self, disparity_image):
        self.disparity_image = disparity_image

    def joint_state_callback(self, joint_state):
        self.pan_eye = joint_state.position[joint_state.name.index("hollie_left_eye_pan_joint")]
        self.tilt_eye = joint_state.position[joint_state.name.index("hollie_eyes_tilt_joint")]
        self.pan_head = joint_state.position[joint_state.name.index("hollie_neck_yaw_joint")]
        self.tilt_head = joint_state.position[joint_state.name.index("hollie_neck_pitch_joint")]

    def link_state_callback(self, link_states):
        self.link_states = link_states

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

    def mem(self, value):
        self.memorize = value.data
        return (True, 'success')

    def transform(self, value):
        point_new = geometry_msgs.msg.PointStamped()
        point_new.header = value.req.header
        point_new.point = value.req.point
        transformed = self.tfBuffer.transform(point_new, self.camera_model.tfFrame())
        transformed_new = PointStamped()
        transformed_new.header = transformed.header
        transformed_new.point = transformed.point
        return transformed_new

def main(args):
    rospy.init_node("head_manager")
    head_manager = HeadManager()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
