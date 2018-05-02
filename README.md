Embodied Attention
==================

This module wraps the deep saliency model developed by Alexander Kroner and the target selection model implemented by Mario Senden in rosnodes, to embody them on a robot.
To use this module on the Neurorobotics Platform, simply add it to your ``GazeboRosPackages/src/`` folder.
You can then copy the launchfile ``nrp.launch`` to your experiment folder and reference it in your ``.exc`` - see the [CDP4 experiment](https://github.com/HBPNeurorobotics/CDP4_experiment) for an example.

Additionally, another rosnode is provided to perform eye movements and save object to memory, when compiled with [holographic](https://github.com/HBPNeurorobotics/holographic).
The integration in ROS has the advantage that the model can be run on a physical robot, such as a pan-tilt unit.

### Dependencies

* tensorflow
* scikit-image

### Testing the model with your webcam

You can use the following launcher:

    roslaunch embodied_attention webcam.launch
