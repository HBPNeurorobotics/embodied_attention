#!/usr/bin/env python

import sys
import os
import numpy as np
from scipy import misc
from collections import OrderedDict

def gauss(x, y, X, Y, sigma):
    return np.exp(-(np.power(x-X, 2)+np.power(y-Y, 2))/(2.*np.power(sigma ,2)))

def positiv(x): return np.maximum(x, 0.)

class Saccade:
    def __init__(self, **kwargs):
        self.N       =  1600       # number of neurons per type (visual, movement)

        default_params = OrderedDict([('sig_lat', 0.122719931),
                                      ('sig_rf', 0.17871822021),
                                      ('sig_IoR', 0.17859038300000002),
                                      ('amp_lat', 0.00029200520000000003),
                                      ('amp_rf', 0.015998337760000002),
                                      ('amp_IoR', 2.42518293),
                                      ('amp_noise', 0.0964379808),
                                      ('k', 0.033419634620000006),
                                      ('g', 0.006138620400000001),
                                      ('theta', 8.2516356),
                                      ('tau', 75.0748425),
                                      ('modulation_type', 'compression'),
                                      ('tau_mod', 75.0748425),
                                      ('sig_mod', 0.17871822021)])
        for key, value in default_params.items():
            setattr(self, key, value)

        self.modulation_type = 'none'
        self.tau_mod = 50.
        self.amp_mod = 1.

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print("Saccade does not have property {}. Can not bet set to value {}.".format(key, value))

        assert(self.modulation_type in ['none', 'shift', 'compression'])

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
        self.motor_neurons += dt*(-self.k*self.motor_neurons + positiv(vis_input - self.g) - self.amp_lat*np.dot(self.W, positiv(self.motor_neurons))) + self.dsig_m*np.random.randn(self.N)

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


        if self.modulation_type is not 'none' and not is_actual_target:
            # fade the modulation towards the multiplicative identity (matrix of ones)
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
