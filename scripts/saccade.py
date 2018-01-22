#!/usr/bin/env python
PKG = 'embodied_attention'
import roslib; roslib.load_manifest(PKG)

import rospy
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from embodied_attention.srv import ResetSaccade
import sys
import os
import numpy as np                                                                                                                                                                                                         
from scipy import misc

## functions
def gauss(x, y, X, Y, sigma):
    return np.exp(-(np.power(x-X, 2)+np.power(y-Y, 2))/(2.*np.power(sigma ,2)))
def f(x): return np.maximum(x, 0.)

class Saccade():
    def __init__(self):
    
        ## parameters
        self.N       =  1600       # number of neurons per type (visual, movement)
        self.theta   =    11.      # decision threshold
        sig_lat      =      .25    # width of Gaussian lateral inhibition
        self.sig_IoR =      .05    # width of Gaussian spread of inhibition of return
        sig_noise    =      .2     # width of Gaussian noise
        self.k       =      .0175  # passive decay rate (movement neurons)
        self.g       =      .33    # input threshold
        self.G       =      .2     # scaling factor for lateral inhibition
        
        ## setup
        # dimensions and coordinate systems
        self.Ns        = int(np.sqrt(self.N))  
        n              = 1./(self.Ns*.5)
        r              = np.linspace(-1., 1., self.Ns)
        self.X, self.Y = np.meshgrid(r, r)
        self.X         = np.reshape(self.X, [self.N, ])
        self.Y         = np.reshape(self.Y, [self.N, ])
        
        # lateral weights 
        self.W        = np.zeros([self.N, self.N])
        for i in range(self.N):
            self.W[:, i] = gauss(self.X[i], self.Y[i], self.X, self.Y, sig_lat)
            self.W[i, i] = 0.
            
        #self.dt = .1
        self.dt = .75
        self.tau = 20.
        
        # noise propagation
        self.dsig_v  = np.sqrt(self.dt/self.tau)*sig_noise # input (visual neuron) noise
        self.dsig_m  = np.sqrt(self.dt)*sig_noise          # movement neuron noise
        
        # (state) variables
        self.M               = np.zeros(self.N)            # movement neurons
        self.V               = np.zeros(self.N)            # visual neurons
    
        self.saliency_sub = rospy.Subscriber("/saliency_map", Float32MultiArray, self.saliency_map_callback, queue_size=1, buff_size=2**24)

        self.target_pub = rospy.Publisher("/saccade_target", Point, queue_size=1)
        self.potential_target_pub = rospy.Publisher("/saccade_potential_target", Point, queue_size=1)

        self.cv_bridge = CvBridge()

        self.reset_saccade_serv = rospy.Service('/reset_saccade', ResetSaccade, self.handle_reset_saccade)

    # numerical integration (simple Euler)
    def saliency_map_callback(self, saliency_map):

        # handle input
        lo = saliency_map.layout
        sal = np.asarray(saliency_map.data[lo.data_offset:]).reshape(lo.dim[0].size, lo.dim[1].size)

        sal = misc.imresize(sal, [self.Ns, self.Ns])
        sal = np.reshape(sal, [self.N, ])/235.*0.55+.2

        # update
        self.V += self.dt*(-self.V + sal)/self.tau + self.dsig_v*np.random.randn(self.N)
        self.M += self.dt*(-self.k*self.M + f(self.V - self.g) - self.G*np.dot(self.W, f(self.M))) + self.dsig_m*np.random.randn(self.N)

        ID = np.argmax(self.M)

        # transform to coordinates in saliency map
        x = np.mod(ID, self.Ns) + 0.5
        y = int(ID/self.Ns) + 0.5
        x_scaled = int(float(lo.dim[0].size)/self.Ns * x)
        y_scaled = int(float(lo.dim[1].size)/self.Ns * y)
        print "potential target: %3d, %3d: %f" % (x_scaled, y_scaled, self.M[ID])

        # puslish potential target
        self.potential_target_pub.publish(Point(x_scaled, y_scaled, self.M[ID]))

        # check if target
        if (self.M[ID] >= self.theta):
            print "\tis target"

            # publish target
            self.target_pub.publish(Point(x_scaled, y_scaled, self.M[ID]))

            # reset
            self.M[ID] = 0.

            # inhibition of return
            self.V = self.V - gauss(self.X[ID], self.Y[ID], self.X, self.Y, self.sig_IoR)

    def handle_reset_saccade(self, req):
        print "going to reset saccade node"
        self.saliency_sub.unregister()
        self.reset_saccade_serv.shutdown()
        self.__init__()
        return True

def main(args):
    rospy.init_node("saccade")
    saccade = Saccade()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
