""" Plot the haemodynamic delay parameter boxplot per region. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess
import itertools
import time
from glob import glob
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.datasets import fetch_atlas_harvard_oxford

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
]


# Global variables
n_regions_to_retain = 10
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

    # python3 boxplot_haemodynamic_delay_per_region.py --plots-dir plots --results-dir results_slrda --verbose 1

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

    columns = ["Group", "sub_tag", "n_run", "lbda", "n_atoms", "a_hat", "ROI"]
    haemo_delays = pd.DataFrame(columns=columns)
    i = 0
    for decomp_dir in decomp_dirs:
        results_path = os.path.join(decomp_dir, 'results.pkl')
        with open(results_path, 'rb') as pfile:
            results = pickle.load(pfile)

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        a_hat = results['a_hat']
        hrf_rois = results['hrf_rois']
        lbda = metadata['decomp_params']['lbda']
        n_atoms = metadata['decomp_params']['n_atoms']
        n_run = metadata['n_run']
        sub_tag = metadata['sub_tag']
        group_label = ("temgesic" if sub_tag_convert[sub_tag] in temgesic_group
                       else "control")

        try:
            for a_hat_, roi_label_ in zip(a_hat, hrf_rois):
                new_row = [group_label, sub_tag, n_run, lbda, n_atoms, a_hat_,
                           roi_label_]
                haemo_delays.loc[i] = new_row
                i += 1

        except KeyError as e:
            continue

    harvard_oxford = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm',
                                                symmetric_split=True)

    n_first_run, n_second_run = np.unique(haemo_delays['n_run']).astype(int)
    unique_n_run = [n_first_run, n_second_run]
    unique_sub_tag = np.unique(haemo_delays['sub_tag'])
    unique_lbdas = np.unique(haemo_delays['lbda'])
    unique_n_atoms = np.unique(haemo_delays['n_atoms']).astype(int)
    param_grid = dict(n_atoms=unique_n_atoms,
                      lbda=unique_lbdas)
    l_plot_params = [dict(zip(param_grid, x))
                     for x in itertools.product(*param_grid.values())
                     ]

    for i, plot_params in enumerate(l_plot_params):
        n_atoms, lbda = plot_params['n_atoms'], plot_params['lbda']

        # Plotting whole brain
        sub_haemo_delays = haemo_delays[(haemo_delays["n_atoms"] == n_atoms) &
                                        (haemo_delays["lbda"] == lbda)]

        plt.figure(f"Figure-{i} (whole brain)", figsize=(5, 5))
        ax = sns.violinplot(x="n_run", y="a_hat", hue="Group",
                            data=sub_haemo_delays, split=True,
                            palette=group_color, width=0.75, linewidth=2.5)
        plt.text(-1.0, 1.05, "Shortest\ndelay", fontsize=14)
        plt.text(-1.0, 0.2, "Longest\ndelay", fontsize=14)
        run_name = [f"Run-{int(t.get_text())}" for t in ax.get_xticklabels()]
        ax.set_xticklabels(run_name, fontdict=dict(fontsize=15))
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlabel("")
        ax.set_ylabel(r"$\delta$", fontsize=20, rotation=0)
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.set_ylim(0.4, 1.0)
        plt.legend(loc='lower center', ncol=2, fontsize=15)
        plt.grid()
        plt.tight_layout()

        filename = (f"boxplot_a_hat_whole_brain_"
                    f"n_atoms_{n_atoms}_lbda_{lbda:.3f}.pdf")
        filepath = os.path.join(args.plots_dir, filename)
        print(f"Saving plot at '{filepath}'")
        plt.savefig(filepath, dpi=200)
        subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

        splitted_fname = filename.split('.')
        filename = '.'.join([splitted_fname[0], splitted_fname[1], 'png'])
        filepath = os.path.join(args.plots_dir, filename)
        print(f"Saving plot at '{filepath}'")
        plt.savefig(filepath, dpi=200)

        plt.clf()
        plt.cla()
        plt.close()

        # Plotting evolution
        sub_haemo_delays = haemo_delays[(haemo_delays["n_atoms"] == n_atoms) &
                                        (haemo_delays["lbda"] == lbda)]

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        xticks = np.arange(len(unique_n_run))
        for sub_tag in np.unique(sub_haemo_delays['sub_tag']):
            filter_sub = sub_haemo_delays['sub_tag'] == sub_tag
            filter_first_run = sub_haemo_delays['n_run'] == n_first_run
            filter_second_run = sub_haemo_delays['n_run'] == n_second_run

            group_label = np.unique(sub_haemo_delays[filter_sub]['Group'])[0]

            filter_sub_first_run = filter_sub & filter_first_run
            filter_sub_second_run = filter_sub & filter_second_run

            a_hat_first_run = sub_haemo_delays[filter_sub_first_run]['a_hat']
            a_hat_second_run = sub_haemo_delays[filter_sub_second_run]['a_hat']

            mean_a_hat = np.array([np.mean(a_hat_first_run),
                                   np.mean(a_hat_second_run)])

            ax.plot(xticks, mean_a_hat, color=group_color[group_label],
                    marker='o', markerfacecolor='tab:gray',
                    markeredgecolor='tab:gray', markersize=2.5, linewidth=2.0)

        ax.set_xticks(xticks)
        run_name = [f"Run-{n_run}" for n_run in unique_n_run]
        ax.set_xticklabels(run_name, fontdict=dict(fontsize=14))
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("")
        ax.set_ylabel(r"$\delta$", fontsize=22, rotation=0)
        ax.yaxis.set_label_coords(-0.25, 0.5)
        ax.set_ylim(0.4, 1.0)
        plt.grid()
        plt.tight_layout()

        filename = (f"evolution_a_hat_whole_brain_"
                    f"n_atoms_{n_atoms}_lbda_{lbda:.3f}.pdf")
        filepath = os.path.join(args.plots_dir, filename)
        print(f"Saving plot at '{filepath}'")
        plt.savefig(filepath, dpi=200)
        subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

        splitted_fname = filename.split('.')
        filename = '.'.join([splitted_fname[0], splitted_fname[1], 'png'])
        filepath = os.path.join(args.plots_dir, filename)
        print(f"Saving plot at '{filepath}'")
        plt.savefig(filepath, dpi=200)

        plt.clf()
        plt.cla()
        plt.close()

        # Plotting per regions
        for n_run in unique_n_run:

            filter = ((haemo_delays["n_atoms"] == n_atoms) &
                      (haemo_delays["lbda"] == lbda) &
                      (haemo_delays["n_run"] == n_run))
            sub_haemo_delays = haemo_delays[filter]

            columns = ['ROI', 'diff_median']
            diff_median_per_regions = pd.DataFrame(columns=columns)
            for m, roi_label in enumerate(np.unique(sub_haemo_delays["ROI"])):
                filter = ((sub_haemo_delays["ROI"] == roi_label) &
                          (sub_haemo_delays["Group"] == 'temgesic'))
                temgesic = sub_haemo_delays[filter]['a_hat']

                filter = ((sub_haemo_delays["ROI"] == roi_label) &
                          (sub_haemo_delays["Group"] == 'control'))
                control = sub_haemo_delays[filter]['a_hat']

                diff_median = np.median(control) - np.median(temgesic)

                diff_median_per_regions.loc[m] = [roi_label, diff_median]

            diff_median_per_regions.sort_values('diff_median', ascending=False,
                                                inplace=True)
            diff_median_per_regions = \
                                diff_median_per_regions[:n_regions_to_retain]

            regions_to_plot = list(diff_median_per_regions["ROI"].astype(int))


            filter = sub_haemo_delays["ROI"].isin(regions_to_plot)
            retained_sub_haemo_delays = sub_haemo_delays[filter]

            plt.figure(f"Figure-{i} (per region)", figsize=(13, 6))
            ax = sns.swarmplot(x="ROI", y="a_hat", hue="Group",
                               data=retained_sub_haemo_delays,
                               order=regions_to_plot, palette=group_color,
                               size=10.0)
            plt.text(-1.1, 1.05, "Shortest\ndelay", fontsize=13)
            plt.text(-1.1, 0.2, "Longest\ndelay", fontsize=13)
            regions_name = [harvard_oxford.labels[m].replace(' ', '\n')
                            for m in regions_to_plot]
            ax.set_xticklabels(regions_name, fontdict=dict(fontsize=12))
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xlabel("")
            ax.set_ylabel(r"$\delta$", fontsize=23, rotation=0)
            ax.yaxis.set_label_coords(-0.05, 0.5)
            ax.set_ylim(0.4, 1.0)
            plt.legend(loc='lower center', ncol=2, fontsize=15)
            plt.grid()
            plt.tight_layout()

            filename = (f"boxplot_a_hat_per_regions_n_run_{n_run}_"
                        f"n_atoms_{n_atoms}_lbda_{lbda:.3f}.pdf")
            filepath = os.path.join(args.plots_dir, filename)
            print(f"Saving plot at '{filepath}'")
            plt.savefig(filepath, dpi=200)
            subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

            splitted_fname = filename.split('.')
            filename = '.'.join([splitted_fname[0], splitted_fname[1], 'png'])
            filepath = os.path.join(args.plots_dir, filename)
            print(f"Saving plot at '{filepath}'")
            plt.savefig(filepath, dpi=200)

            plt.clf()
            plt.cla()
            plt.close()


    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
