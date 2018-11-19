from scipy import misc
import scipy.io as sio
import argparse
from attention import Saliency
from attention import Saccade
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from os import path
import cma
from collections import OrderedDict
from tqdm import tqdm
import datetime
import pickle

parser = argparse.ArgumentParser(description='Test saliency model')
parser.add_argument('--model', type=str, required=True,
                    help='path to the model.ckpt file')
parser.add_argument('--fixations', type=str, required=True,
                    help='path to the SALICON fixations')
parser.add_argument('--images', type=str, required=True,
                    help='path to the SALICON images')
parser.add_argument('--out', type=str, default='/tmp/salicon',
                    help='path to output folder')
parser.add_argument('--n-train', type=int, default=10,
                    help='number of SALICON image used to compute the fitness')
parser.add_argument('--n-iter', type=int, default=50,
                    help='number of CMAES iterations')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed')

args = parser.parse_args()

try:
    os.makedirs(args.out)
except OSError as e:
    print(e)

model_file = args.model
saliency = Saliency(model_file=model_file)
fixation = sio.loadmat(args.fixations)
all_user_fixations = [ len(fixation['gaze'][i][0][2]) for i in range(len(fixation['gaze'])) ]
time_limit = fixation["gaze"][0][0][1][-1][0]
gt_mean_rate = np.mean(all_user_fixations) * 1000. / time_limit
path_to_image = os.path.join(args.images, fixation['image'][0] + '.jpg')
print("Ground truth image {} has mean saccade rate: {}, std: {} ({} participants)"
      .format(path_to_image, gt_mean_rate, np.std(all_user_fixations), len(fixation['gaze'])))
image = misc.imread(path_to_image)
saliency_map = saliency.compute_saliency_map(image)
dt = 5.

# ordered list of params to optimize
saccade_params = OrderedDict([
    ('sig_lat', .1),
    ('sig_rf', 0.267),
    ('sig_IoR', .1),
    ('amp_lat', .001),
    ('amp_rf', 0.008),
    ('amp_IoR', 1.5),
    ('amp_noise', .09),
    ('k', .017),
    ('g', .33),
    ('theta', 6.),
    ('tau', 50.)
])


optimize_params = [
    'sig_lat',
    'sig_rf',
    'sig_IoR',
    'amp_lat',
    'amp_rf',
    'amp_IoR',
    'amp_noise',
    'k',
    'g',
    'theta',
    'tau'
]


# convert the cmaes params to the full state x by inserting static params
def x_to_params(param_scaling):
    params = saccade_params.copy()
    for i, p in enumerate(optimize_params):
        params[p] *= param_scaling[i]
    return params


def simulate(sol, n_it, i):
    print(datetime.datetime.now())
    params = x_to_params(sol)
    np.random.seed(args.seed)
    saccade = Saccade(**params)
    time = 0
    all_targets = []
    all_times = []
    while time < time_limit:
        (target, is_actual_target) = saccade.compute_saccade_target(saliency_map, dt=dt)
        if is_actual_target:
            all_targets.append(target)
            all_times.append(time)
        time += dt
    rate = len(all_targets) * 1000. / time_limit
    error = gt_mean_rate - rate
    print("Solution {} has rate: {}. Error: {}".format(params, rate, error))
    return error

# get list of indices of the params we are optimizing
optimize_params_idx = np.array([ saccade_params.keys().index(param) for param in optimize_params ])
initial_dyn_params = np.array(saccade_params.values())[optimize_params_idx]
# we perform optimization on scaling factors for each param
startmean = np.ones(len(optimize_params))
# scale bounds accordingly
# when original param value is negative, we need to change the order of the inequality (the bounds)
sigma0 = 0.2
bound_factors = [0, 2.5]
bounds = [
    bound_factors[0] * np.ones(len(optimize_params)),
    bound_factors[1] * np.ones(len(optimize_params))
]
n_iter = args.n_iter
popsize = np.floor(len(bounds[0]) * 2. / 3.)

with open(os.path.join(args.out, 'initial.pkl'), 'w') as f:
    pickle.dump({
        'initial_dyn_params': initial_dyn_params,
        'optimize_params': optimize_params,
        'sigma0': sigma0,
        'bounds': bound_factors,
        'n_iter': n_iter,
        'popsize': popsize
    }, f)


es = cma.CMAEvolutionStrategy(x0=startmean,
                              sigma0=sigma0,
                              inopts={'seed':args.seed, 'bounds' : bounds, 'maxiter': n_iter ,'popsize': popsize})
es.logger.name_prefix = os.path.join(args.out, "outcmaes")

n_it = 0


while not es.stop():
    solutions = es.ask()
    errors = []
    for (i, sol) in enumerate(tqdm(solutions)):
        err = simulate(sol, n_it, i)
        errors.append(err)
    es.tell(solutions, errors)
    es.logger.add()  # write data to disc to be plotted
    es.disp()
    es.result_pretty()
    print(es.best.x)
    #cma.plot()  # shortcut for es.logger.plot()
    n_it += 1
