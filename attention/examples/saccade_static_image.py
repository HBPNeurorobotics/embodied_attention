from scipy import misc
import argparse
from attention import Saliency
from attention import Saccade
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from os import path

parser = argparse.ArgumentParser(description='Test saliency model')
parser.add_argument('--model', type=str, required=True,
                    help='path to the model.ckpt file')
parser.add_argument('--out', type=str, default='/tmp/saliency',
                    help='path to output folder')
parser.add_argument('--image', type=str, required=True,
                    help='path to the input image')

args = parser.parse_args()

try:
    os.makedirs(args.out)
except OSError as e:
    print(e)

# demonstration of saliency model
model_file = args.model
saliency = Saliency(model_file=model_file)
image = misc.imread(args.image)
saliency_map = saliency.compute_saliency_map(image)
plt.imsave(path.join(args.out, 'saliency.png'), saliency_map, cmap=plt.get_cmap('gray'))

# demonstration of saccade model
saccade = Saccade()
ims = []
dt = 5
simulation_time = 1000
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('visual neurons')
ax2.set_title('motor neurons')

for k in range(simulation_time / dt):
    (target, is_actual_target) = saccade.compute_saccade_target(saliency_map, dt=dt)
    visual_neurons = saccade.visual_neurons.reshape(saccade.Ns, saccade.Ns)
    motor_neurons = saccade.motor_neurons.reshape(saccade.Ns, saccade.Ns)
    ims.append( [
        ax1.imshow(visual_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none'),
        ax2.imshow(motor_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none')
    ])

ani = animation.ArtistAnimation(fig, ims, interval=dt, blit=True)
ani.save(path.join(args.out, 'neurons.mp4'))
