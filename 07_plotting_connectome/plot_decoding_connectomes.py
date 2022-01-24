""" Plot the feature importance for the decoding on z-maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


# Main
if __name__ == '__main__':

    # python3 plot_decoding_connectomes.py --connectomes-dir ../04_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 3 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--connectomes-dir', type=str,
                        default='results_connectome',
                        help='Set the name of the connectomes directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--seed', type=int, default=None, help='Seed.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the seed base anlaysis z-maps
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_connectome_paths = os.path.join(args.connectomes_dir,
                                             f"corr_matrice_sub-*_run-*_"
                                             f"tr-*_group-*.npy")
    connectome_paths = glob(template_connectome_paths)

    runs = []
    columns = ['sub', 'run', 'group', 'connectome']
    connectomes = pd.DataFrame(columns=columns)
    for i, connectome_path in enumerate(connectome_paths):

        connectome_path = os.path.normpath(connectome_path)
        connectome_name = os.path.basename(connectome_path)
        chunks = connectome_name.split('_')
        sub, run, group = chunks[2], chunks[3], chunks[5]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1].split('.')[0]
        group = 0 if group == 'control' else 1

        connectome = np.load(connectome_path)
        connectome = connectome[np.tril(connectome, -1) == 0].flatten()

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(connectome_paths):02d}] Connectome"
                  f" extraction for '{connectome_path}'", end='')

        connectomes.loc[i] = [sub, run, group, connectome]

        runs.append(run)

    print()

    runs = np.unique(runs)

    ###########################################################################
    # Fit the decoder model
    i = 0
    scoring = 'accuracy'
    scores = pd.DataFrame(columns=['run', 'mean-score', 'std-score'])
    best_mean_test_scores, best_std_test_score = [], []

    for i, run in enumerate(runs):

        filter = connectomes['run'] == run

        X = list(connectomes[filter]['connectome'])
        y = list(connectomes[filter]['group'])

        X, y = np.array(X), np.array(y)

        if args.verbose:
            print(f"[{run}] Fitting decoder...")

        grid = GridSearchCV(
                    # estimator=LogisticRegression(),
                    # param_grid=[dict(C=list(np.logspace(-10, 10, 21)),
                    #                  fit_intercept=[False, True])],
                    # estimator=RidgeClassifier(),
                    # param_grid=[dict(alpha=list(np.logspace(-10, 10, 21)),
                    #                  fit_intercept=[False, True])],
                    estimator=SVC(),
                    param_grid=[dict(C=list(np.logspace(-10, 10, 21)),
                                     kernel=['linear', 'rbf'])],
                    cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=20),
                    scoring='accuracy',
                    n_jobs=args.cpu,
                                ).fit(X, y)

        mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]
        std_score = grid.cv_results_["std_test_score"][grid.best_index_]

        if args.verbose:
            print(f"[{run}] Best score = {mean_score:.2f} "
                  f"+/- {std_score/2:.2f}")

        scores.loc[i] = [int(run.split('-')[-1]), mean_score, std_score]

        if args.verbose:
            print(f"[{run}] Best parameters: "
                  f"{pprint.pformat(grid.best_params_)}")

    ###########################################################################
    # Plotting
    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.errorbar(scores['run'], scores['mean-score'], scores['std-score'],
                fmt='o', capsize=2., markersize=2.5, color='tab:blue',
                ecolor='tab:blue', lw=1.5)
    for y, lw in zip([0.0, 0.5, 1.0], [.5, 1., .5]):
        plt.axhline(y, lw=lw, color='gray')
    for x in 0.5 + np.array(scores['run']):
        plt.axvline(x, lw=0.5, color='gray')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0.25, 1.)
    fig.tight_layout()
    filename = os.path.join(args.plots_dir, f"accuracy_boxplot")

    if args.verbose:
        print(f"Saving plot under '{filename + '.pdf'}'")
    fig.savefig(filename + '.pdf', dpi=300)

    if args.verbose:
        print(f"Saving plot under '{filename + '.png'}'")
    fig.savefig(filename + '.png', dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
