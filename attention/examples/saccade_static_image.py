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
parser.add_argument('--rf_modulation_type', type=str, default='none',
                    help='type of receptive field modulation')


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
# plt.imsave(path.join(args.out, 'saliency.png'), saliency_map, cmap=plt.get_cmap('gray'))

# demonstration of saccade model
def plot_targets(all_targets, all_times, saliency_map, out):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(saliency_map,
              vmin=0, vmax=1.,
              aspect='equal',
              interpolation='none',
              cmap=plt.get_cmap('gray'),
              origin='upper')
    ax.set_axis_off()
    all_targets = np.array(all_targets).T
    if len(all_targets) > 0:
        ax.scatter(all_targets[0], all_targets[1], c=all_times, s=50, cmap=plt.get_cmap('Reds'))
    plt.savefig(path.join(out, 'targets.png'), dpi=150)

def draw_rf(ax, rf):
    return ax.imshow(np.max(rf, axis=0),
              aspect='equal', cmap=plt.get_cmap('Reds'),
              interpolation='bilinear',
              vmin=0, vmax=1.)

saccade = Saccade(modulation_type=args.rf_modulation_type)
ims = []
dt = 5
simulation_time = 1000
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title('visual neurons')
ax2.set_title('motor neurons')
ax3.set_title('receptive fields')

all_targets = []
all_times = []

time = 0
neuron_min = -3
neuron_max = saccade.theta
epsilon = 5
rf_ids = np.array([
    saccade.Ns * (saccade.Ns / 2 - epsilon) + saccade.Ns / 2 - epsilon,
    saccade.Ns * (saccade.Ns / 2 - epsilon) + saccade.Ns / 2 + epsilon,
    saccade.Ns * (saccade.Ns / 2 + epsilon) + saccade.Ns / 2 - epsilon,
    saccade.Ns * (saccade.Ns / 2 + epsilon) + saccade.Ns / 2 + epsilon])

for k in range(simulation_time / dt):
    (target, is_actual_target) = saccade.compute_saccade_target(saliency_map, dt=dt)
    time += dt
    if is_actual_target:
        all_targets.append(target)
        all_times.append(time)
    visual_neurons = saccade.visual_neurons.reshape(saccade.Ns, saccade.Ns).copy()
    motor_neurons = saccade.motor_neurons.reshape(saccade.Ns, saccade.Ns).copy()
    receptive_fields = saccade.receptive_fields[rf_ids].reshape(-1, saccade.Ns, saccade.Ns)
    modulation = saccade.modulation[rf_ids].reshape(-1, saccade.Ns, saccade.Ns)
    modulated_receptive_fields = np.multiply(receptive_fields, modulation)
    ims.append( [
        ax1.imshow(visual_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none', vmin=neuron_min, vmax=neuron_max),
        ax2.imshow(motor_neurons, animated=True, aspect='equal', cmap=plt.get_cmap('gray'), interpolation='none', vmin=neuron_min, vmax=neuron_max),
        draw_rf(ax3, modulated_receptive_fields)
    ])
print("{} <= visual neuron <= {}\n{} <= motor neuron <= {}"
      .format(visual_neurons.min(), visual_neurons.max(),
              motor_neurons.min(), motor_neurons.max()
      ))
print("Performed {} saccades in {}ms. Rate: {}"
      .format(len(all_times), simulation_time, len(all_times)* 1000. / simulation_time))

ani = animation.ArtistAnimation(fig, ims, interval=10*dt, blit=True)
ani.save(path.join(args.out, 'neurons.mp4'))
plot_targets(all_targets, all_times, saliency_map, out=args.out)
