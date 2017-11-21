Embodied Attention
==================

This module wraps the deep saliency model developed by Alexander Kroner in a rosnode, to embody it on a robot.
To use this module on the Neurorobotics Platform, add it to your ``GazeboRosPackages/`` folder.

Additionally, another rosnode is provided to perform eye movements and save object to memory, when compiled with [holographic](https://github.com/HBPNeurorobotics/holographic).
The integration in ROS has the advantage that the model can be run on a physical robot, such as a pan-tilt unit.

### Dependencies

* keras==1.2.2
* theano==0.9.0
* scikit-image
* wget (used to download the weights/topology of the saliency network on first run)

### Launching the model

Use the following launcher:

    roslaunch embodied_attention nrp.launch
