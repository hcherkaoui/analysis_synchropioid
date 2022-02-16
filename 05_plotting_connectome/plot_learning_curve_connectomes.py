""" Plot the feature importance for the decoding on z-maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, learning_curve


# Main
if __name__ == '__main__':

    # python3 plot_learning_curve_connectomes.py --connectomes-dir ../03_connectome/results_connectome/ --plots-dir plots --seed 0 --cpu 3 --task-filter only_hb_rest --verbose 1

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
    # Collect the seed base anlaysis z-maps
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_connectome_paths = os.path.join(args.connectomes_dir,
                                             f"corr_matrice_sub-*_run-*_"
                                             f"tr-*_group-*.npy")
    connectome_paths = glob(template_connectome_paths)

    i = 0
    columns = ['group', 'run', 'connectome']
    connectomes = pd.DataFrame(columns=columns)
    for connectome_path in connectome_paths:

        connectome_path = os.path.normpath(connectome_path)
        connectome_name = os.path.basename(connectome_path)
        chunks = connectome_name.split('_')

        run = f"Run-{int(chunks[3].split('-')[-1])}"
        group = chunks[5].split('-')[-1].split('.')[0]
        group = 0 if group == 'control' else 1
        t_r = float(chunks[4].split('-')[-1])

        connectome = np.load(connectome_path)
        connectome = connectome[np.tril(connectome, -1) == 0].flatten()

        if t_r in valid_tr:

            if args.verbose:
                print(f"\r[{i+1:02d}/{len(connectome_paths):02d}] Connectome"
                    f" extraction for '{connectome_path}'", end='')

            connectomes.loc[i] = [group, run, connectome]

            i += 1

    print()

    ###########################################################################
    # Fit the decoder model
    scoring = 'accuracy'
    ratios = np.linspace(0.3, 1.0, 16)

    filter = connectomes['run'] != 'Run-1'

    X = list(connectomes[filter]['connectome'])
    y = list(connectomes[filter]['group'])

    X, y = np.array(X), np.array(y)

    model = SVC(C=1e-10, kernel='linear')
    res = learning_curve(model,
                    X, y, train_sizes=ratios,
                    cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=40),
                    scoring=scoring, n_jobs=args.cpu, shuffle=True,
                    random_state=args.seed, verbose=args.verbose)

    train_sizes_abs, train_scores, test_scores = res

    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)

    ###########################################################################
    # Plotting
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(train_sizes_abs, mean_train_scores, lw=1.0, label='train',
            color="tab:green")
    ax.fill_between(train_sizes_abs,
                    mean_train_scores - std_train_scores / 2.0,
                    mean_train_scores + std_train_scores / 2.0,
                    alpha=0.1, color="tab:green")
    ax.plot(train_sizes_abs, mean_test_scores, lw=1.0, label='test',
            color="tab:orange")
    ax.fill_between(train_sizes_abs,
                    mean_test_scores - std_test_scores / 2.0,
                    mean_test_scores + std_test_scores / 2.0,
                    alpha=0.1, color="tab:orange")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0.5, 1.0)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    filename = os.path.join(args.plots_dir, f"accuracy_learning_curve")

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
