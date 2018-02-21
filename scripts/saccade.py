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

class SaccadeNode():
    def __init__(self):
        self.saccade = Saccade()

        self.saliency_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)

        self.saccade_target_pub = rospy.Publisher("/saccade_target", Point, queue_size=1)
        self.saccade_potential_target_pub = rospy.Publisher("/saccade_potential_target", Point, queue_size=1)

        self.reset_saccade_serv = rospy.Service('/reset_saccade', ResetSaccade, self.handle_reset_saccade)

        self.last_time = rospy.get_time()

    def saliency_map_callback(self, saliency_map):
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time

        lo = saliency_map.layout
        saliency_map = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

        (target, is_actual_target) = self.saccade.compute_saccade_target(saliency_map, dt)

        self.saccade_potential_target_pub.publish(Point(*target))
        if is_actual_target:
            self.saccade_target_pub.publish(Point(*target))

    def handle_reset_saccade(self, req):
        rospy.loginfo("Resetting node")
        self.saccade = Saccade()
        return True

def main():
    rospy.init_node("saccade")
    saccade_node = SaccadeNode()
    rospy.spin()

if __name__ == "__main__":
    main()
