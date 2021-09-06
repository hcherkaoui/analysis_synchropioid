""" Plot the haemodynamic delay parameter boxplot per region. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess
import time
from glob import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Global variables
temgesic_group = ['S00634', 'S00651', 'S00669', 'S00726', 'S00748', 'S00791',
                  'S00805']
group_color = dict(control='tab:blue', temgesic='tab:orange')
participants_fname = '../data/nifti_dir/participants.tsv'
participants_path = os.path.abspath(os.path.normpath(participants_fname))
participants = pd.read_csv(participants_path, sep='\t')
sub_tag_convert = dict(zip(participants['participant_id'],
                           participants['DICOM tag']))


# Main
if __name__ == '__main__':

    # python3 connectome_comparison_subjects.py --plots-dir plots --results-dir ../02_connectome/results_connectome/ --verbose 1

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
    results_paths = set([npy_fname for npy_fname in
                         glob(f"{results_dir}/**/*.npy", recursive=True)])

    if len(results_paths) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    columns = ["Group", "sub_tag", "n_run", "det"]
    connectomes = pd.DataFrame(columns=columns)
    for i, results_path in enumerate(results_paths):

        run_dir_path = os.path.dirname(results_path)
        sub_dir_path = os.path.dirname(run_dir_path)

        connectome = np.load(results_path)
        det = np.linalg.det(connectome)
        n_run = int(os.path.basename(run_dir_path)[-1])
        sub_tag = os.path.basename(sub_dir_path)
        group_label = ("temgesic" if sub_tag_convert[sub_tag] in temgesic_group
                       else "control")

        connectomes.loc[i] = [group_label, sub_tag, n_run, det]

    unique_all_n_run = np.unique(connectomes['n_run'])
    unique_sub_tag = np.unique(connectomes['sub_tag'])

    # Plotting evolution
    fig, ax = plt.subplots(1, 1, figsize=(9, 2))
    for sub_tag in unique_sub_tag:
        filter_sub = connectomes['sub_tag'] == sub_tag

        group_label = np.unique(connectomes[filter_sub]['Group'])[0]

        unique_n_run = np.unique(connectomes[filter_sub]['n_run'])
        unique_n_run.astype(int).sort()

        l_det = []
        for n_run in unique_n_run:
            filter_run = connectomes['n_run'] == n_run
            filter_sub_run = filter_sub & filter_run
            l_det.append(float(connectomes[filter_sub_run]['det']))

        ax.plot(unique_n_run, l_det, color=group_color[group_label],
                marker='o', markerfacecolor='tab:gray',
                markeredgecolor='tab:gray', markersize=2.5, linewidth=2.0)

    ax.set_xticks(unique_all_n_run)
    run_name = [f"Run-{n_run}" for n_run in unique_all_n_run]
    ax.set_xticklabels(run_name, fontdict=dict(fontsize=14))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Det(conn)", fontsize=14, rotation=90)
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()

    filename = (f"evolution_connectome.pdf")
    filepath = os.path.join(args.plots_dir, filename)
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=200)
    subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

    splitted_fname = filename.split('.')
    filename = '.'.join([splitted_fname[0], 'png'])
    filepath = os.path.join(args.plots_dir, filename)
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=200)

    plt.clf()
    plt.cla()
    plt.close()

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
