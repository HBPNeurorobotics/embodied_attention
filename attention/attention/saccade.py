#!/usr/bin/env python

import sys
import os
import numpy as np
from scipy import misc

def gauss(x, y, X, Y, sigma):
    return np.exp(-(np.power(x-X, 2)+np.power(y-Y, 2))/(2.*np.power(sigma ,2)))

def positiv(x): return np.maximum(x, 0.)

class Saccade:
    def __init__(self, modulation_type='none',
                 amp_rf=0.008, sig_rf=0.267,
                 amp_mod=1., sig_mod=0.267):
        ## parameters taken from:
        ## Purcell et al. (2012). From Salience to Saccades: Multiple-Alternative Gated Stochastic Accumulator Model of Visual Search. Journal of Neuroscience, 32(10)
        self.N       =  1600       # number of neurons per type (visual, movement)
        self.theta   =    6.   # decision threshold
        self.sig_lat      = .1     # width of Gaussian lateral inhibition
        self.sig_rf      =  sig_rf # width of Gaussian receptive field
        self.amp_rf      =  amp_rf # scaling factor for receptive field
        self.sig_mod  =  sig_mod   # width of Gaussian modulation
        self.amp_mod  =  amp_mod   # amplitude of Gaussian modulation
        self.sig_IoR =      .1    # width of Gaussian spread of inhibition of return
        self.amp_IoR =      1.5     # strength of inhibition of return
        self.amp_noise =    .09    # strength of noise
        self.k       =      .017   # passive decay rate (movement neurons)
        self.g       =      .33    # input threshold
        self.G       =      .001    # scaling factor for lateral inhibition
        self.modulation_type = modulation_type
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
        self.receptive_fields = np.zeros([self.N, self.N])
        self.original_receptive_fields = np.zeros([self.N, self.N])
        self.modulation = np.ones([self.N, self.N])
        for i in range(self.N):
            self.W[:, i] = gauss(self.X[i], self.Y[i], self.X, self.Y, self.sig_lat)
            self.W[i, i] = 0.
            self.receptive_fields[:, i] = gauss(self.X[i], self.Y[i], self.X, self.Y, self.sig_rf)
            self.original_receptive_fields[:, i] = self.receptive_fields[:, i].copy()

        self.tau = 50.
        self.tau_mod = 50.

        # (state) variables
        self.visual_neurons = self.amp_noise*np.random.randn(self.N) # visual neurons
        self.motor_neurons  = self.amp_noise*np.random.randn(self.N) # movement neurons

        self.last_winner = None

    # numerical integration (simple Euler)
    def compute_saccade_target(self, saliency_map, dt=30.):
        self.dsig_v = np.sqrt(dt/self.tau)*self.amp_noise # input (visual neuron) noise
        self.dsig_m = np.sqrt(dt)*self.amp_noise          # movement neuron noise

        sal = misc.imresize(saliency_map, [self.Ns, self.Ns])
        sal = np.reshape(sal, [self.N, ])/235.*0.55+.2

        # update
        self.visual_neurons += dt*(-self.visual_neurons + sal)/self.tau + self.dsig_v*np.random.randn(self.N)
        vis_input = self.amp_rf * np.dot(np.multiply(self.receptive_fields, self.modulation), self.visual_neurons)
        self.motor_neurons += dt*(-self.k*self.motor_neurons + positiv(vis_input - self.g) - self.G*np.dot(self.W, positiv(self.motor_neurons))) + self.dsig_m*np.random.randn(self.N)

        ID = np.argmax(self.motor_neurons)

        # transform to coordinates in saliency map
        y = int(ID/self.Ns) + 0.5
        x = np.mod(ID, self.Ns) + 0.5
        y_scaled = int(float(len(saliency_map))/self.Ns * y)
        x_scaled = int(float(len(saliency_map[0]))/self.Ns * x)
        # print("Winning neuron: x: %3d, y: %3d, value: %f" % (x_scaled, y_scaled, self.motor_neurons[ID]))

        target = (x_scaled, y_scaled, self.motor_neurons[ID])
        is_actual_target = False

        # check if target
        if (self.motor_neurons[ID] >= self.theta):
            is_actual_target = True
            self.last_winner = ID

            # reset
            self.motor_neurons[ID] = 0.

            # inhibition of return
            self.visual_neurons -= self.amp_IoR * gauss(self.X[ID], self.Y[ID], self.X, self.Y, self.sig_IoR)

            # receptive field modulation
            if self.modulation_type == 'shift':
                for i in range(self.N):
                    self.modulation[:, i] = self.amp_mod * gauss(self.X[ID]+self.X[i],self.Y[ID]+self.Y[i],self.X,self.Y, self.sig_mod)
            elif self.modulation_type == 'compression':
                for i in range(self.N):
                    self.modulation[:, i] = self.amp_mod * gauss(self.X[ID],self.Y[ID],self.X,self.Y, self.sig_mod)

        # fade the modulation towards np.ones()
        self.modulation += dt*(-self.modulation  + np.ones([self.N, self.N]))/self.tau_mod

        return (target, is_actual_target)

    def shift(self):
        # shift activity
        if self.last_winner is not None:
            print("\tShifting activity")
            dy = self.Ns/2 - int(self.last_winner/self.Ns)
            dx = self.Ns/2 - np.mod(self.last_winner, self.Ns)
            visual_neurons = np.reshape(self.visual_neurons, [self.Ns, self.Ns])
            motor_neurons = np.reshape(self.motor_neurons, [self.Ns, self.Ns])
            if dy > 0:
                visual_neurons = np.pad(visual_neurons, ((dy,0),(0,0)), mode='constant')[:-dy,:]
                motor_neurons = np.pad(motor_neurons, ((dy,0),(0,0)), mode='constant')[:-dy,:]
            else:
                visual_neurons = np.pad(visual_neurons, ((0,-dy),(0,0)), mode='constant')[-dy:,:]
                motor_neurons = np.pad(motor_neurons, ((0,-dy),(0,0)), mode='constant')[-dy:,:]
            if dx > 0:
                visual_neurons = np.pad(visual_neurons, ((0,0),(dx,0)), mode='constant')[:,:-dx]
                motor_neurons = np.pad(motor_neurons, ((0,0),(dx,0)), mode='constant')[:,:-dx]
            else:
                visual_neurons = np.pad(visual_neurons, ((0,0),(0,-dx)), mode='constant')[:,-dx:]
                motor_neurons = np.pad(motor_neurons, ((0,0),(0,-dx)), mode='constant')[:,-dx:]
            self.visual_neurons = visual_neurons.flatten()
            self.motor_neurons = motor_neurons.flatten()
