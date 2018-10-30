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
parser.add_argument('--image', type=str, required=True,
                    help='path to the input image')
parser.add_argument('--out', type=str, default='/tmp/saccade',
                    help='path to output folder')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed')

parser.add_argument('--n-amp', type=int, default=10,
                    help='number of amplitudes to try')
parser.add_argument('--n-std', type=int, default=10,
                    help='number of standard deviations to try')
parser.add_argument('--min-amp', type=float, default=0.0005,
                    help='minimum amplitude of the gaussian receptive fields')
parser.add_argument('--max-amp', type=float, default=0.05,
                    help='maximum amplitude of the gaussian receptive fields')
parser.add_argument('--min-std', type=float, default=0.1,
                    help='minimum standard deviation of the gaussian receptive fields')
parser.add_argument('--max-std', type=float, default=0.5,
                    help='maximum standard deviation of the gaussian receptive fields')

args = parser.parse_args()
rf_stds = np.linspace(args.min_std, args.max_std, args.n_std)
rf_amps = np.linspace(args.min_amp, args.max_amp, args.n_amp)

try:
    os.makedirs(args.out)
except OSError as e:
    print(e)

def plot_first_spike_rf(rf_amps, rf_stds, first_saccades, out):
    x, y = np.meshgrid(rf_amps, rf_stds)
    human_reference=0.317 * 1000.
    first_saccades = first_saccades - human_reference
    time_limit = np.max(first_saccades)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.contourf(x, y, first_saccades, cmap=plt.get_cmap('seismic'))
    cbar = fig.colorbar(cax, ticks=[0])
    cbar.ax.set_yticklabels(['human reference'])
    ax.set_xlabel('RF strength')
    ax.set_ylabel('RF width')
    plt.savefig(path.join(out, 'first_saccades.png'), dpi=150)

def plot_rf(rf, std, out):
    ax.imshow(rf,
              aspect='equal', cmap=plt.get_cmap('plasma'),
              interpolation='bilinear',
              vmin=0, vmax=1.)
    plt.savefig(path.join(out, 'rf_std_{:.3f}.png'.format(std)), dpi=150)


model_file = args.model
saliency = Saliency(model_file=model_file)
image = misc.imread(args.image)
saliency_map = saliency.compute_saliency_map(image)
plt.imsave(path.join(args.out, 'saliency.png'), saliency_map, cmap=plt.get_cmap('gray'))

dt = 1.
first_saccades = np.zeros((rf_amps.size, rf_stds.size))
time_limit = 800.
np.random.seed(args.seed)
fig = plt.figure()
for i, amp in enumerate(rf_amps):
    for j, std in enumerate(rf_stds):
        saccade = Saccade(amp_rf=amp, sig_rf=std)
        ax = fig.add_subplot(111)
        if i == 0:
            rf_id = (saccade.Ns + 1) * saccade.Ns / 2
            plot_rf(saccade.receptive_fields[rf_id].reshape(saccade.Ns, saccade.Ns), std, args.out)

        is_actual_target = False
        time = 0
        while not is_actual_target and time < time_limit:
            (target, is_actual_target) = saccade.compute_saccade_target(saliency_map, dt=dt)
            time += dt
        print("receptive field with amp {:.3f} std {:.3f} took {}ms".format(amp, std, int(time)))
        first_saccades[i, j] = time

np.save(path.join(args.out, 'results.npy'), {
    'rf_amps': rf_amps,
    'rf_stds': rf_stds,
    'first_saccades': first_saccades
})


plot_first_spike_rf(rf_amps, rf_stds, first_saccades, args.out)
