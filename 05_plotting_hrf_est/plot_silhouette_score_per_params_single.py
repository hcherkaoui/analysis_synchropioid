""" Plot the mean haemodynamic delay parameter for each SLRDA decompostion
parameters. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess
import time
from glob import glob
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}'
]


# Main
if __name__ == '__main__':

    # python3 plot_silhouette_score_per_params_single.py --plots-dir plots --results-dir ../02_hrf_est/results_hrf_estimation/ --task-filter only_hb_rest --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--task-filter', type=str,
                        default='all_task',
                        help='Filter the fMRI task loaded, valid options are '
                             '["only_hb_rest", "only_rest", "all_task"].')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.task_filter == 'only_hb_rest':
        valid_tr = [0.8]

    elif args.task_filter == 'only_rest':
        valid_tr = [2.0]

    elif args.task_filter == 'all_task':
        valid_tr = [0.8, 2.0]

    else:
        valid_tr = [0.8, 2.0]

    ###########################################################################
    # Collect functional data
    if args.verbose:
        print(f"Creating '{args.plots_dir}'")

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    decomp_dirs = [os.path.dirname(pickle_fname) for pickle_fname in
                   glob(f"{results_dir}/**/*.pkl", recursive=True)]
    decomp_dirs = set(decomp_dirs)
    decomp_dirs = list(decomp_dirs)

    ###########################################################################
    # Extract useful data
    if len(decomp_dirs) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    i = 0
    columns = ['sub', 'run', 'group', 'lbda', 'a_hat']
    haemo_delays = pd.DataFrame(columns=columns)
    for n, decomp_dir in enumerate(decomp_dirs):

        results_path = os.path.join(decomp_dir, 'results.pkl')
        with open(results_path, 'rb') as pfile:
            results = pickle.load(pfile)

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        try:
            lbda = metadata['decomp_params']['lbda']
            run = metadata['n_run']
            t_r = metadata['t_r']
            group = 0 if metadata['group_label'] == 'control' else 1
            sub_tag = metadata['sub_tag']

            if t_r in valid_tr:

                for a_hat_ in results['a_hat']:
                    new_col = [sub_tag, run, group, lbda, a_hat_]
                    haemo_delays.loc[i] = new_col
                    i += 1

        except KeyError as e:
            continue

        if args.verbose:
            print(f"\r[{n+1:02d}/{len(decomp_dirs):02d}] Extracting results "
                  f"from '{os.path.basename(decomp_dir)}'", end='')

    ###########################################################################
    # Compute useful statistic
    unique_lbdas = np.unique(haemo_delays['lbda'])
    n_unique_lbdas = len(unique_lbdas)
    lbdas, mean_a_hat, std_a_hat, silhouette_score = [], [], [], []

    best_score, best_params = -1., dict(lbda=-1.)
    i, n_params_unique = 0, n_unique_lbdas
    for m, lbda in enumerate(unique_lbdas):

        filter = haemo_delays['lbda'] == lbda

        sub_haemo_delays = haemo_delays[filter]

        a_hat = sub_haemo_delays['a_hat']

        mean_a_hat.append(np.mean(a_hat))
        std_a_hat.append(np.std(a_hat))
        lbdas.append(lbda)

        unique_sub = np.unique(sub_haemo_delays['sub'])
        unique_run = np.unique(sub_haemo_delays['run'])

        X, Y = [], []
        for sub in unique_sub:
            for run in unique_run:

                filter = ((sub_haemo_delays['sub'] == sub) &
                          (sub_haemo_delays['run'] == run))

                x = sub_haemo_delays[filter]['a_hat']
                y = sub_haemo_delays[filter]['group']

                if len(y) != 0:  # check if subject does not have this run
                    X.append(np.array(x))
                    Y.append(int(np.unique(y)))

        X, Y = np.r_[X], np.r_[Y]

        score = metrics.silhouette_score(X, Y)
        silhouette_score.append(score)

        if score > best_score:
            best_score = score
            best_params['lbda'] = float(lbda)

        if args.verbose:
            print(f"\r[{i+1:02d}/{n_params_unique:02d}] Processing "
                  f"statistics", end='')

        i += 1

    ###########################################################################
    # Send back best params
    decomp_params_dir = 'decomp_params'
    if not os.path.isdir(decomp_params_dir):
        os.makedirs(decomp_params_dir)

    jsonfname = 'best_single_subject_decomp_params.json'
    jsonfname = os.path.join(decomp_params_dir, jsonfname)

    if args.verbose:
        print(f"Saving best parameters under '{jsonfname}'")
    with open(jsonfname, 'w') as jsonfile:
        json.dump(best_params, jsonfile)

    ###########################################################################
    # Plotting
    selected_lbda = lbdas[np.argmax(silhouette_score)]

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 4))

    axs[0].plot(lbdas, silhouette_score, marker='.',
                color='tab:orange')
    axs[0].axvline(selected_lbda, lw=0.75, c='k')
    axs[0].text(selected_lbda, -0.4, rf"${selected_lbda:.4f}$")
    axs[0].grid()
    axs[0].set_xscale('log')
    axs[0].set_ylabel("silhouette\nscore", fontsize=11, rotation=90)
    axs[0].set_ylim(-0.2, 0.2)

    axs[1].errorbar(lbdas, mean_a_hat, yerr=std_a_hat, marker='.',
                    color='tab:blue')
    axs[1].axvline(selected_lbda, lw=0.75, c='k')
    axs[1].text(selected_lbda, -0.4, rf"${selected_lbda:.4f}$")
    axs[1].grid()
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r"$\lambda_f$", fontsize=14)
    axs[1].set_ylabel(r"$\overline{\delta}$", fontsize=14, rotation=0)
    axs[1].set_ylim(0.2 - 0.1, 2.0 + 0.1)

    plt.tight_layout()

    filename = os.path.join(args.plots_dir,
                            'silhouette_score_per_params_single')

    print(f"Saving plot at '{filename + '.pdf'}'")
    plt.savefig(filename + '.pdf', dpi=300)
    subprocess.call(f"pdfcrop {filename+'.pdf'} {filename+'.pdf'}", shell=True)

    print(f"Saving plot at '{filename + '.png'}'")
    plt.savefig(filename + '.png', dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
