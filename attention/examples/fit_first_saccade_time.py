from scipy import misc
import argparse
from attention import Saliency
from attention import Saccade
import numpy as np
import matplotlib
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
parser.add_argument('--min-amp', type=float, default=0.005,
                    help='minimum amplitude of the gaussian receptive fields')
parser.add_argument('--max-amp', type=float, default=0.011,
                    help='maximum amplitude of the gaussian receptive fields')
parser.add_argument('--min-std', type=float, default=0.1,
                    help='minimum standard deviation of the gaussian receptive fields')
parser.add_argument('--max-std', type=float, default=0.35,
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

    fastest = np.min(first_saccades)
    first_saccades = first_saccades - human_reference
    time_limit = np.max(first_saccades)
    min_limit = np.min(first_saccades)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks = np.array([100, 200, 400, 500, 600, 700])
    levels = np.arange(-human_reference, np.max(ticks), 30)
    cax = ax.contourf(x, y, first_saccades, levels=levels, cmap=plt.get_cmap('RdYlGn'),
                      vmin=-time_limit, vmax=time_limit
    )
    human_idx = np.searchsorted(ticks, human_reference)
    ticks = np.insert(ticks, human_idx, human_reference)
    ticks_labels = ticks.astype(str)

    cbar = fig.colorbar(cax, ticks=ticks - human_reference)
    cbar.ax.set_yticklabels(ticks_labels)
    cbar.set_label('Time to first saccade (ms)')

    ax.set_xlabel('RF strength')
    ax.set_ylabel('RF width')
    plt.savefig(path.join(out, 'first_saccades.png'), dpi=150)

def plot_rf(rf, std, out):
    ax = fig.add_subplot(111)
    ax.imshow(rf,
              aspect='equal', cmap=plt.get_cmap('Reds'),
              interpolation='bilinear',
              vmin=0, vmax=1.)
    plt.savefig(path.join(out, 'rf_std_{:.3f}.png'.format(std)), dpi=150)

def plot_targets_rf(all_targets, saliency_map, out):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(saliency_map,
              vmin=0, vmax=1.,
              aspect='equal',
              interpolation='bilinear',
              cmap=plt.get_cmap('gray'),
              origin='upper')
    ax.set_axis_off()
    cmap = matplotlib.cm.get_cmap('Reds')
    n = len(all_targets.keys())
    for i, std in enumerate(sorted(all_targets.keys())):
        rf_targets = np.array(all_targets[std]).T
        if len(rf_targets) is not 0:
            rgba = cmap(float(i) / (n - 1))
            ax.scatter(rf_targets[0], rf_targets[1], c=rgba, s=50)

    plt.savefig(path.join(out, 'targets.png'), dpi=150)

model_file = args.model
saliency = Saliency(model_file=model_file)
image = misc.imread(args.image)
saliency_map = saliency.compute_saliency_map(image)
plt.imsave(path.join(args.out, 'saliency.png'), saliency_map, cmap=plt.get_cmap('gray'))

dt = 5.
first_saccades = np.zeros((rf_amps.size, rf_stds.size))
all_targets = { std: [] for std in rf_stds }
time_limit = 800.
fig = plt.figure()
for i, amp in enumerate(rf_amps):
    for j, std in enumerate(rf_stds):
        np.random.seed(args.seed)
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
        print("receptive field with amp {:.3f} std {:.3f} took {}ms. {} <= motor <= {}"
              .format(amp, std, int(time), saccade.motor_neurons.min(), saccade.motor_neurons.max()))
        first_saccades[i, j] = time
        if is_actual_target:
            all_targets[std].append(target)

np.save(path.join(args.out, 'results.npy'), {
    'saliency_map': saliency_map,
    'image': image,
    'rf_amps': rf_amps,
    'rf_stds': rf_stds,
    'first_saccades': first_saccades,
    'all_targets': all_targets
})


plot_first_spike_rf(rf_amps, rf_stds, first_saccades, args.out)
plot_targets_rf(all_targets, saliency_map, args.out)
