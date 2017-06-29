Embodied Attention
==================

Saliency model that publishes joint states for the FLIR-PTU-D64 by focusing on interesting objects viewed by a Kinect mounted on top of the unit.

### Dependencies

Clone the OpenCV Vision repository from Github:

    git clone https://github.com/ros-perception/vision_opencv.git

Switch to the branch matching your ROS distribution release.
    
### Setting up FLIR-PTU-D64

Follow these instructions:

http://ids-wiki.fzi.de/index.php/HBP/DVS-head

### Mounting Kinect

Place the kinect on the PTU, and bind it, for example with tape, so that it doesn't move. Make sure to have enough wiggle room for it's USB cable.

### Launching the model

Use the following launcher:

    roslaunch saliency kinect.launch 
    
### Notes

The weights used in the saliency model are placed in the following folder:

    /disk/no_backup/jkaiser/saliency/

Should these files move their path needs to be updated in the forward.py script.