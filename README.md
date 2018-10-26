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
You can download the model on HBP SP10 owncloud:

```bash
download_saliency () {
    if [ $1 ]; then
        curl -k -o model.ckpt.meta "https://neurorobotics-files.net/owncloud/index.php/s/hdjl7TjzSUqF1Ww/download"
        curl -k -o model.ckpt.index "https://neurorobotics-files.net/owncloud/index.php/s/DCPB80foqkteuC4/download"
        curl -k -o model.ckpt.data-00000-of-00001 "https://neurorobotics-files.net/owncloud/index.php/s/bkpmmvrVkeELapr/download"
        echo "gpu" > config
    else
        curl -k -o model.ckpt.meta "https://neurorobotics-files.net/owncloud/index.php/s/TNpWFSX8xLvfbYD/download"
        curl -k -o model.ckpt.index "https://neurorobotics-files.net/owncloud/index.php/s/sDCFUGTrzJyhDA5/download"
        curl -k -o model.ckpt.data-00000-of-00001 "https://neurorobotics-files.net/owncloud/index.php/s/Scti429S7D11tMv/download"
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

Then in your python script:

```python
from attention import Saliency
from attention import Saccade
import numpy as np
import matplotlib.pyplot as plt

# demonstration of saliency model
model_file = "path/to/model.cpkt" # the TensorFlow model files you downloaded from HBP SP10 owncloud
saliency = Saliency(model_file=model_file)
image = np.random.randn(200,200,3)
saliency_map = saliency.compute_saliency_map(image)
plt.imsave('/tmp/saliency.png', saliency_map)

# demonstration of saccade model
saccade = Saccade()
(target, is_actual_target, visual_neurons, motor_neurons) = saccade.compute_saccade_target(saliency_map, dt=1000)
```
