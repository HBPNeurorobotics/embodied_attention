import argparse
from attention import Saliency
from attention import Saccade
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description='Test saliency model')
parser.add_argument('--model', type=str,
                    help='path to the model.ckpt file')
args = parser.parse_args()


# demonstration of saliency model
model_file = args.model
saliency = Saliency(model_file=model_file)
image = np.random.randn(200,200,3)
saliency_map = saliency.compute_saliency_map(image)
plt.imsave('/tmp/saliency.png', saliency_map)

# demonstration of saccade model
saccade = Saccade()
ims = []
dt = 30
simulation_time = 2000
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('visual neurons')
ax2.set_title('motor neurons')

for k in range(simulation_time / dt):
    (target, is_actual_target, visual_neurons, motor_neurons) = saccade.compute_saccade_target(saliency_map, dt=dt)
    ims.append( [
        ax1.imshow(visual_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none'),
        ax2.imshow(motor_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none')
    ])

ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True,
                                repeat_delay=1000)
ani.save('/tmp/saccade_neurons.mp4')
