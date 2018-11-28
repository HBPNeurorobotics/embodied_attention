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
from skimage.draw import circle

parser = argparse.ArgumentParser(description='Test saliency model')
parser.add_argument('--model', type=str, required=True,
                    help='path to the model.ckpt file')
parser.add_argument('--image', type=str, required=True, nargs='+',
                    help='path to the input image')
parser.add_argument('--out', type=str, default='/tmp/fit_saccade_targets',
                    help='path to output folder')
parser.add_argument('--n-iter', type=int, default=200,
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
images = [ misc.imread(im) for im in args.image ]
saliency_maps = [ saliency.compute_saliency_map(im) for im in images ]
for (i, sal) in enumerate(saliency_maps):
    plt.imsave(path.join(args.out, 'saliency_{}.png'.format(i)), sal, cmap=plt.get_cmap('gray'))
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


def simulate(sol, n_it, i, time_limit=5000):
    params = x_to_params(sol)
    all_errors = []
    all_rates = []
    for sal in saliency_maps:
        np.random.seed(args.seed)
        saccade = Saccade(**params)
        time = 0
        all_times = []
        score_img = np.zeros_like(sal)
        radius_rf = 5
        while time < time_limit:
            (target, is_actual_target) = saccade.compute_saccade_target(sal, dt=dt)
            if is_actual_target:
                rr, cc = circle(target[1], target[0], 5, shape=score_img.shape)
                score_img[rr, cc] = sal[rr, cc]
                all_times.append(time)
            time += dt
        rate = len(all_times) * 1000. / time_limit
        # we want score to be big with few saccades
        if len(all_times) == 0:
            # strongly penalize when no saccades are made
            reg = 100
        else:
            reg = rate

        error = - score_img.sum() + np.exp(reg *0.5)
        all_errors.append(error)
        all_rates.append(rate)
    print("Solution {} has rates: {}. Error: {}".format(params, all_rates, sum(all_errors)))
    return sum(all_errors)

# get list of indices of the params we are optimizing
optimize_params_idx = np.array([ saccade_params.keys().index(param) for param in optimize_params ])
initial_dyn_params = np.array(saccade_params.values())[optimize_params_idx]
# we perform optimization on scaling factors for each param
startmean = np.ones(len(optimize_params))
# scale bounds accordingly
# when original param value is negative, we need to change the order of the inequality (the bounds)
sigma0 = 0.2
bound_factors = [0, 2.]
bounds = [
    bound_factors[0] * np.ones(len(optimize_params)),
    bound_factors[1] * np.ones(len(optimize_params))
]
n_iter = args.n_iter
popsize = np.floor(len(optimize_params) * 2. / 3.)

with open(os.path.join(args.out, 'initial.pkl'), 'w') as f:
    pickle.dump({
        'initial_dyn_params': initial_dyn_params,
        'saccade_params': saccade_params,
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
    print("Iteration {} starting at time {}".format(n_it, datetime.datetime.now()))
    for (i, sol) in enumerate(tqdm(solutions)):
        err = simulate(sol, n_it, i)
        errors.append(err)
    es.tell(solutions, errors)
    es.logger.add()  # write data to disc to be plotted
    es.disp()
    es.result_pretty()
    # print(es.best.x)
    #cma.plot()  # shortcut for es.logger.plot()
    n_it += 1

print("Optimization finished. Data was recorded in {}".format(args.out))

def plot_fitting_resuts(cmaes_path):
    import snn_utils.visualization as viz
    import matplotlib.pyplot as plt
    import pickle
    import os
    initial = pickle.load( open( os.path.join(cmaes_path, "initial.pkl"), "rb" ) )
    ax = viz.plot_cmaes_cost(cmaes_path)
    ax.set_ylim(-5000, 1000)
    plt.savefig(os.path.join(cmaes_path, 'cmaes_cost.png'))
    ax = viz.plot_cmaes_params(cmaes_path,
                               labels=initial['optimize_params'])
    plt.savefig(os.path.join(cmaes_path, 'cmaes_results.png'))
    ax.legend()
    best, fitness = viz.get_cmaes_best(cmaes_path)
    print("Best ever: Fitness {} solution: {}".format(fitness, best))

try:
    print("Attempting to plot results")
    plot_fitting_resuts(args.out)
except Exception as e:
    print("Results could not be plotted: {}".format(e))
