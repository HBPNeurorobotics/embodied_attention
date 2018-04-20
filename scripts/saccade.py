#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
from attention import Saccade
from embodied_attention.srv import ResetSaccade
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty

class SaccadeNode():
    def __init__(self):
        self.saccade = Saccade()

        self.saliency_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)

        self.saccade_target_pub = rospy.Publisher("/saccade_target", Point, queue_size=1)
        self.saccade_potential_target_pub = rospy.Publisher("/saccade_potential_target", Point, queue_size=1)
        self.visual_neurons_pub = rospy.Publisher("/visual_neurons", Image, queue_size=1)
        self.motor_neurons_pub = rospy.Publisher("/motor_neurons", Image, queue_size=1)

        self.reset_saccade_serv = rospy.Service('/reset_saccade', ResetSaccade, self.handle_reset_saccade)
        self.shift_serv = rospy.Service('/shift', Empty, self.handle_shift)

        self.last_time = None

        self.cv_bridge = CvBridge()

        self.saliency_map = None

    def saliency_map_callback(self, saliency_map):
        rospy.loginfo("Saliency_map updated")
        lo = saliency_map.layout
        self.saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

    def handle_reset_saccade(self, req):
        rospy.loginfo("Resetting node")
        self.saccade = Saccade()
        return True

    def handle_shift(self, req):
        rospy.loginfo("Shifting activity")
        self.saccade.shift()
        return

    def loop(self):
        hz = 1000. # 10000 would be perfect
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            if self.saliency_map is None:
                continue

            (target, is_actual_target, visual_neurons, motor_neurons) = self.saccade.compute_saccade_target(self.saliency_map, 1./hz * 1000.)

            self.saccade_potential_target_pub.publish(Point(*target))
            if is_actual_target:
                self.saccade_target_pub.publish(Point(*target))

            visual_neurons = (visual_neurons - visual_neurons.min()) / (visual_neurons.max() - visual_neurons.min())
            motor_neurons = (motor_neurons - motor_neurons.min()) / (motor_neurons.max() - motor_neurons.min())

            try:
                visual_neurons_image = self.cv_bridge.cv2_to_imgmsg(np.uint8(visual_neurons * 255.), "mono8")
                motor_neurons_image = self.cv_bridge.cv2_to_imgmsg(np.uint8(motor_neurons * 255.), "mono8")
            except CvBridgeError as e:
                print e

            self.visual_neurons_pub.publish(visual_neurons_image)
            self.motor_neurons_pub.publish(motor_neurons_image)

            rate.sleep()


def main():
    rospy.init_node("saccade")
    saccade_node = SaccadeNode()
    saccade_node.loop()

if __name__ == "__main__":
    main()
