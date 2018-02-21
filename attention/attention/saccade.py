#!/usr/bin/env python

import sys
import os
import numpy as np
from scipy import misc

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
    
    # numerical integration (simple Euler)
    def compute_saccade_target(self, saliency_map):
        sal = misc.imresize(saliency_map, [self.Ns, self.Ns])
        sal = np.reshape(sal, [self.N, ])/235.*0.55+.2

        # update
        self.V += self.dt*(-self.V + sal)/self.tau + self.dsig_v*np.random.randn(self.N)
        self.M += self.dt*(-self.k*self.M + f(self.V - self.g) - self.G*np.dot(self.W, f(self.M))) + self.dsig_m*np.random.randn(self.N)

        ID = np.argmax(self.M)

        # transform to coordinates in saliency map
        x = np.mod(ID, self.Ns) + 0.5
        y = int(ID/self.Ns) + 0.5
        x_scaled = int(float(len(saliency_map))/self.Ns * x)
        y_scaled = int(float(len(saliency_map[0]))/self.Ns * y)
        print("Potential target: %3d, %3d: %f" % (x_scaled, y_scaled, self.M[ID]))

        target = (x_scaled, y_scaled, self.M[ID])
        is_actual_target = False

        # check if target
        if (self.M[ID] >= self.theta):
            print("\tis actual target")

            is_actual_target = True

            # reset
            self.M[ID] = 0.

            # inhibition of return
            self.V = self.V - gauss(self.X[ID], self.Y[ID], self.X, self.Y, self.sig_IoR)

        return (target, is_actual_target)
