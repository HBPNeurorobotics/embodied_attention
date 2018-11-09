Embodied Attention
==================

This module wraps two models from Maastricht University:
- the deep saliency model developed by Alexander Kroner
- the target selection model implemented by Mario Senden.

Additionally, another rosnode is provided to perform eye movements and save object to memory, when compiled with [holographic](https://github.com/HBPNeurorobotics/holographic).
The integration in ROS has the advantage that the model can be run on a physical robot, such as a pan-tilt unit.

### Dependencies

* tensorflow
* scikit-image

### Installing

If you use the Neurorobotics, this module installs itself with the install script provided in [CDP4 experiment](https://github.com/HBPNeurorobotics/CDP4_experiment).

Otherwise, there are multiple ways you can use the models provided in this package.
In any way, you will need the files defining the TensorFlow model for the saliency network.
You can download the model on HBP SP10 nextcloud:

```bash
download_saliency () {
    if [ $1 ]; then
        curl -k -o model.ckpt.meta "https://neurorobotics-files.net/index.php/s/SNsfXBjm3bpNoMB/download"
        curl -k -o model.ckpt.index "https://neurorobotics-files.net/index.php/s/tZ64sd9diA2X8QZdownload"
        curl -k -o model.ckpt.data-00000-of-00001 "https://neurorobotics-files.net/index.php/s/nHtn8Q9ezxLtP3B/download"
        echo "gpu" > config
    else
        curl -k -o model.ckpt.meta "https://neurorobotics-files.net/index.php/s/B3mB7aRKpy6EEE2/download"
        curl -k -o model.ckpt.index "https://neurorobotics-files.net/index.php/s/zXW99pqBmKN2TCf/download"
        curl -k -o model.ckpt.data-00000-of-00001 "https://neurorobotics-files.net/index.php/s/X5g6ajtSNefYpnH/download"
        echo "cpu" > config
    fi
}
 # optional argument: gpu
download_saliency
```

#### On the Neurorobotics Platform

To use this module on the Neurorobotics Platform, simply add it to your ``GazeboRosPackages/src/`` folder.
You can then copy the launchfile ``nrp.launch`` to your experiment folder and reference it in your ``.exc`` - see the [CDP4 experiment](https://github.com/HBPNeurorobotics/CDP4_experiment) for an example.

#### As a rosnode

You can use the following launcher to test the model with your webcam using ROS:

    roslaunch embodied_attention webcam.launch

#### Python standalone

You can install the provided python package ```attention``` to use the models in your python script the following way.

```bash
cd attention
pip install -e .
```

Have a look to [attention/examples/saccade_static_image.py](attention/examples/saccade_static_image.py) for an example script on how to use the saliency and saccade modules.
