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
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}'
]


# Main
if __name__ == '__main__':

    # python3 mean_haemodynamic_delay_per_params.py --plots-dir plots --results-dir results_slrda --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.verbose:
        print(f"Creating '{args.plots_dir}'")

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    decomp_dirs = set([os.path.dirname(pickle_fname) for pickle_fname in
                       glob(f"{results_dir}/**/*.pkl", recursive=True)])

    if len(decomp_dirs) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    haemo_delays = pd.DataFrame(columns=['lbda', 'n_atoms', 'a_hat'])
    i = 0
    for decomp_dir in decomp_dirs:
        results_path = os.path.join(decomp_dir, 'results.pkl')
        with open(results_path, 'rb') as pfile:
            results = pickle.load(pfile)

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        try:
            lbda = metadata['decomp_params']['lbda']
            n_atoms = metadata['decomp_params']['n_atoms']
            for a_hat_ in results['a_hat']:
                haemo_delays.loc[i] = [lbda, n_atoms, a_hat_]
                i += 1

        except KeyError as e:
            continue

    unique_lbdas = np.unique(haemo_delays['lbda'])
    unique_n_atoms = np.unique(haemo_delays['n_atoms']).astype(int)

    n_unique_lbdas, n_unique_n_atoms = len(unique_lbdas), len(unique_n_atoms)
    mean_heatmap = np.empty(shape=(n_unique_lbdas, n_unique_n_atoms))
    std_heatmap = np.empty(shape=(n_unique_lbdas, n_unique_n_atoms))

    for m, lbda in enumerate(unique_lbdas):
        for n, n_atoms in enumerate(unique_n_atoms):
            filter = ((haemo_delays['lbda'] == lbda) |
                      (haemo_delays['n_atoms'] == n_atoms))
            sub_haemo_delays = haemo_delays[filter]
            a_hat = sub_haemo_delays['a_hat']
            mean_heatmap[m, n] = np.mean(a_hat)
            std_heatmap[m, n] = np.std(a_hat)

    plt.figure("Haemodynamic delays", figsize=(4, 3))
    ax = sns.heatmap(mean_heatmap, annot=std_heatmap,
                     xticklabels=unique_n_atoms, yticklabels=unique_lbdas)
    plt.text(3.5, 0.2, "Shortest\ndelay", fontsize=9)
    plt.text(3.5, 3.25, "Longest\ndelay", fontsize=9)
    plt.xlabel("K", fontsize=15)
    plt.ylabel(r"$\lambda_f$", fontsize=15, rotation=0)
    ax.yaxis.set_label_coords(-0.2, 0.5)
    plt.tight_layout()
    filename = os.path.join(args.plots_dir, 'haemodynamic_delays.pdf')
    print(f"Saving plot at '{filename}'")
    plt.savefig(filename, dpi=200)
    subprocess.call(f"pdfcrop {filename} {filename}", shell=True)
    filename = os.path.join(args.plots_dir, 'haemodynamic_delays.png')
    print(f"Saving plot at '{filename}'")
    plt.savefig(filename, dpi=200)

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
