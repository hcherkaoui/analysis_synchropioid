""" Plot the haemodynamic delay parameter boxplot per region. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess
import time
from glob import glob
import argparse
import pickle
import json
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from hemolearn import utils


temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


# Main
if __name__ == '__main__':

    # python3 plot_shared_maps.py --results-dir ../03_hrf_est/results_hrf_estimation_group/ --best-params-file best_group_decomp_params.json --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--best-params-file', type=str,
                        default='best_group_decomp_params.json',
                        help='Load the best decomposition parameters.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the functional data
    with open(args.best_params_file, 'r') as jsonfile:
        best_params = json.load(jsonfile)

    if args.verbose:
            print(f"Best parameters ({best_params}) loaded from "
                  f"'{args.best_params_file}'")

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    decomp_dirs = set([os.path.dirname(pickle_fname) for pickle_fname in
                       glob(f"{results_dir}/**/*.pkl", recursive=True)])

    if len(decomp_dirs) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    ###########################################################################
    # Compute statistic
    retained_dirs = dict(control=None, temgesic=None)
    for decomp_dir in decomp_dirs:

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        if retained_dirs[metadata['group_label']] is None:
            retained_dirs[metadata['group_label']] = decomp_dir

        if all([v is not None for v in retained_dirs.values()]):
            break

    results_path = os.path.join(retained_dirs['control'], 'results.pkl')
    with open(results_path, 'rb') as pfile:
        results_control = pickle.load(pfile)

    results_path = os.path.join(retained_dirs['temgesic'], 'results.pkl')
    with open(results_path, 'rb') as pfile:
        results_temgesic = pickle.load(pfile)

    u_hat_control = results_control['u_hat']
    u_hat_temgesic = results_temgesic['u_hat']

    cost = u_hat_control.dot(u_hat_temgesic.T)
    row_ind, col_ind = linear_sum_assignment(cost)

    shared_maps_control = [results_control['u_hat_img'][i] for i in row_ind]
    shared_maps_temgesic = [results_temgesic['u_hat_img'][i] for i in col_ind]

    ###########################################################################
    # Plotting
    nrows, ncols = len(shared_maps_control), 2
    _, axis = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 3))

    for i in range(nrows):
        plotting.plot_stat_map(
                        shared_maps_control[i], axes=axis[i, 0],
                        threshold=utils.th(shared_maps_control[i].get_fdata(),
                                           t='5%', absolute=False),
                        bg_img=datasets.MNI152_FILE_PATH
                        )

        plotting.plot_stat_map(
                        shared_maps_temgesic[i], axes=axis[i, 1],
                        threshold=utils.th(shared_maps_temgesic[i].get_fdata(),
                                           t='5%', absolute=False),
                        bg_img=datasets.MNI152_FILE_PATH
                        )

    filepath = os.path.join(args.plots_dir, 'shared_maps')

    print(f"Saving plot under '{filepath + '.png'}'")
    plt.savefig(filepath + '.png', dpi=150)

    print(f"Saving plot under '{filepath + '.pdf'}'")
    plt.savefig(filepath + '.pdf', dpi=150)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
