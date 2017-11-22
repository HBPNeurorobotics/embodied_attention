Embodied Attention
==================

This module wraps the deep saliency model developed by Alexander Kroner in a rosnode, to embody it on a robot.
To use this module on the Neurorobotics Platform, simply add it to your ``GazeboRosPackages/`` folder.
You can then copy the launchfile ``nrp.launch`` to your experiment folder and reference it in your ``.exc`` - see the [CDP4 experiment](https://github.com/HBPNeurorobotics/CDP4_experiment) for an example.

Additionally, another rosnode is provided to perform eye movements and save object to memory, when compiled with [holographic](https://github.com/HBPNeurorobotics/holographic).
The integration in ROS has the advantage that the model can be run on a physical robot, such as a pan-tilt unit.

### Dependencies

* keras==1.2.2
* theano==0.9.0
* scikit-image
* h5py
* wget (used to download the weights/topology of the saliency network on first run)

### Testing the model with your webcam

You can use the following launcher:

    roslaunch embodied_attention webcam.launch
