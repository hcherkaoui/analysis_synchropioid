""" Plot the feature importance for the decoding on z-maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
from nilearn import input_data, plotting, image


# set fontsize
sns.set(font_scale=0.7)


# Main
if __name__ == '__main__':

    # python3 plot_connectome_norm_evolution.py --connectome-dir ../04_connectome/results_connectome/ --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--connectome-dir', type=str, default='output_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the connectomes
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_connectome_paths = os.path.join(args.connectome_dir,
                                             f"corr_matrice_sub-*_run-*_"
                                             f"tr-*_group-*.npy")
    connectome_paths = glob(template_connectome_paths)

    runs = []
    columns = ['sub', 'run', 'group', 'connectome-mean']
    connectomes = pd.DataFrame(columns=columns)
    for i, connectome_path in enumerate(connectome_paths):

        connectome_path = os.path.normpath(connectome_path)
        connectome_name = os.path.basename(connectome_path)
        chunks = connectome_name.split('_')
        sub, run, group = chunks[2], chunks[3], chunks[5]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1].split('.')[0]

        connectome_mean = np.mean(np.load(connectome_path))

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(connectome_paths):02d}] Connectome"
                  f" extraction for '{connectome_path}'", end='')

        connectomes.loc[i] = [sub, run, group, connectome_mean]

        runs.append(run)

    print()

    connectomes = connectomes[(np.abs(stats.zscore(connectomes['connectome-mean'])) < 3)]
    runs = np.unique(runs)

    ###########################################################################
    # Plotting
    palette = {"control":'tab:blue',
               "temgesic":'tab:orange'}
    order = ["Run-1", "Run-2", "Run-3", "Run-4", "Run-5"]

    plt.figure('Boxplot', figsize=(3, 4))
    ax = sns.stripplot(data=connectomes, x='run', y='connectome-mean',
                       hue='group', palette=palette, order=order, jitter=.2,
                       size=4., linewidth=.75, alpha=0.75)
    for x in 0.5 + np.arange(5):
        plt.axvline(x, lw=1., color='gray')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .8), fontsize=5,
              ncol=3)
    plt.tight_layout()
    filename = os.path.join(args.plots_dir, f"mean_connectivity_boxplot")

    if args.verbose:
        print(f"Saving plot under '{filename + '.pdf'}'")
    plt.savefig(filename + '.pdf', dpi=300)

    if args.verbose:
        print(f"Saving plot under '{filename + '.png'}'")
    plt.savefig(filename + '.png', dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
