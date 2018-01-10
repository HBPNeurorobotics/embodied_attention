#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from embodied_attention.srv import Roi, Look
from ros_holographic.srv import NewObject, ProbeLabel
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np

class Attention():
    def __init__(self):
        saccade_sub = rospy.Subscriber("/saccade_target", Point, self.saccade_callback, queue_size=1, buff_size=2**24)
        camera_sub = rospy.Subscriber("/icub_model/left_eye_camera/image_raw", Image, self.camera_callback, queue_size=1, buff_size=2**24)
        camera_info_sub = rospy.Subscriber("/icub_model/left_eye_camera/camera_info", CameraInfo, self.camera_info_callback, queue_size=1, buff_size=2**24)

        self.pan_pub = rospy.Publisher("/robot/left_eye_pan/pos", Float64, queue_size=1)
        self.tilt_pub = rospy.Publisher("/robot/eye_tilt/pos", Float64, queue_size=1)
        self.roi_pub = rospy.Publisher("/roi", Image, queue_size=1)

        self.recognizer = rospy.ServiceProxy('recognize', Roi)
        self.memory = rospy.ServiceProxy('new_object', NewObject)
        self.probe = rospy.ServiceProxy('probe_label', ProbeLabel)

        self.looker = rospy.Service('look', Look, self.look)

        self.camera = None
        self.camera_info = None
        self.model = PinholeCameraModel()

        self.x = 0.
        self.y = 0.

        self.limit = 0.942477796 

        self.saliency_width = float(rospy.get_param('~saliency_width', '256'))
        self.saliency_height = float(rospy.get_param('~saliency_height', '192'))

        self.cv_bridge = CvBridge()

    def saccade_callback(self, saccade):
        if self.camera is not None and self.camera_info is not None:
            # scale to camera image size
            x = int(saccade.x * (float(self.camera.width)/self.saliency_width))
            y = int(saccade.y * (float(self.camera.height)/self.saliency_height))

            # compute target in polar coordinates
            self.model.fromCameraInfo(self.camera_info)
            ray = self.model.projectPixelTo3dRay((x, y))
            dx = np.arctan(-ray[0])
            dy = np.arctan(-ray[1] / np.sqrt(ray[0] * ray[0] + 1))

            # publish new position
            if self.x + dx < -self.limit or self.x + dx > self.limit or self.y + dy < -self.limit or self.y + dy > self.limit:
                print "\tover eye movement limit! dropped!"
                return
            self.x += dx
            self.y += dy
            self.pan_pub.publish(self.x)
            self.tilt_pub.publish(self.y)

            # set roi
            size = 25
            x1 = self.camera_info.width/2 - size
            y1 = self.camera_info.height/2 - size
            x2 = self.camera_info.width/2 + size
            y2 = self.camera_info.height/2 + size

            try:
                image = self.cv_bridge.imgmsg_to_cv2(self.camera, "bgr8")
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
                try:
                    # store in memory
                    self.memory(self.x * 100, self.y * 100, label)
                    print "\tstored in memory: %d, %d, %s" % (self.x * 100, self.y * 100, label)
                except rospy.ServiceException:
                    print "\tmemory service call failed"
            except rospy.ServiceException:
                print "\trecognize service call failed"

        else:
            print "but information is missing"

    def camera_callback(self, camera):
        self.camera = camera

    def camera_info_callback(self, camera_info):
        self.camera_info = camera_info

    def look(self, label):
        # ask memory for label
        p = self.probe(label.Label)

        if not p.return_value or len(p.X) == 0 or len(p.Y) == 0:
            return False

        x = p.X[0]
        y = p.Y[0]

        # adjust camera
        self.pan_pub.publish(x/100.)
        self.tilt_pub.publish(y/100.)
        return True

def main(args):
    rospy.init_node("attention")
    attention = Attention()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
