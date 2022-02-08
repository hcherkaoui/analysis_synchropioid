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
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
]


temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


# Main
if __name__ == '__main__':

    # python3 plot_haemodynamic_delays_comparison_subjects.py --plots-dir plots --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir/ --results-dir ../02_hrf_est/results_hrf_estimation/ --best-params-file decomp_params/best_single_subject_decomp_params.json --task-filter only_hb_rest --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--bids-root-dir', type=str,
                        default='fmri_nifti_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--n-regions-to-retain', type=int, default=10,
                        help='Number of regions to display.')
    parser.add_argument('--best-params-file', type=str,
                        default='best_group_decomp_params.json',
                        help='Load the best decomposition parameters.')
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
    # Collect the functional data
    participants_fname = os.path.join(args.bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                               participants['DICOM tag']))

    with open(args.best_params_file, 'r') as jsonfile:
        best_params = json.load(jsonfile)

    if args.verbose:
            print(f"Best parameters ({best_params}) loaded from "
                  f"'{args.best_params_file}'")

    if args.verbose:
        print(f"Saving connectomes plots under '{args.plots_dir}' directory")

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    results_dir = os.path.abspath(os.path.normpath(args.results_dir))
    decomp_dirs = set([os.path.dirname(pickle_fname) for pickle_fname in
                       glob(f"{results_dir}/**/*.pkl", recursive=True)])

    if len(decomp_dirs) == 0:
        raise ValueError(f"No results found under '{args.results_dir}'")

    ###########################################################################
    # Compute statistic
    columns = ['sub', 'group', 't_r', 'run', 'a_hat', 'roi']
    haemo_delays = pd.DataFrame(columns=columns)
    runs, i = [], 0
    for decomp_dir in decomp_dirs:
        results_path = os.path.join(decomp_dir, 'results.pkl')
        with open(results_path, 'rb') as pfile:
            results = pickle.load(pfile)

        metadata_path = os.path.join(decomp_dir, 'metadata.json')
        with open(metadata_path, 'r') as jsonfile:
            metadata = json.load(jsonfile)

        try:

            a_hat = results['a_hat']
            hrf_rois = results['hrf_rois']

            t_r = metadata['t_r']
            lbda = metadata['decomp_params']['lbda']
            n_atoms = metadata['decomp_params']['n_atoms']
            run = metadata['n_run']
            sub = (metadata['sub_tag'] if 'sub_tag' in metadata
                                       else metadata['sub_tags'])
            group = metadata['group_label']

            shared_maps = True if isinstance(a_hat, list) else False

            if lbda == best_params['lbda']:

                if t_r in valid_tr:

                    if shared_maps:
                        for sub_, a_hat_ in zip(sub, a_hat):
                            for a_hat__, roi_label_ in zip(a_hat_, hrf_rois):
                                new_row = [sub_, group, t_r, run, a_hat__,
                                        roi_label_]
                                haemo_delays.loc[i] = new_row
                                i += 1

                    else:
                        for a_hat_, roi_label_ in zip(a_hat, hrf_rois):
                            new_row = [sub, group, t_r, run, a_hat_, roi_label_]
                            haemo_delays.loc[i] = new_row
                            i += 1

                runs.append(run)

        except KeyError as e:
            continue

    all_runs = np.sort(np.unique(runs))

    ###########################################################################
    # Plotting
    group_color = dict(control='tab:blue', temgesic='tab:orange')
    markers = list(Line2D.markers.keys())
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))

    for group_label in ['control', 'temgesic']:

        if group_label == 'temgesic':

            offset = 0.2
            color = 'tab:orange'

            filter = haemo_delays['group'] == 'temgesic'
            sub_temgesics = np.unique(haemo_delays[filter]['sub'])

            for marker, sub in zip(markers, sub_temgesics):

                filter = haemo_delays['sub'] == sub

                x = haemo_delays[filter]['run']
                y = haemo_delays[filter]['a_hat']

                jitter = 0.05 * np.random.randn(*x.shape)

                plt.scatter(x + jitter + offset, y, alpha=0.2, s=5.0,
                            marker=marker, color=color)

        else:

            offset = -0.2
            color = 'tab:blue'

            filter = haemo_delays['group'] == 'control'
            sub_controls = np.unique(haemo_delays[filter]['sub'])

            for marker, sub in zip(markers, sub_controls):

                x = haemo_delays[filter]['run']
                y = haemo_delays[filter]['a_hat']

                jitter = 0.05 * np.random.randn(*x.shape)

                plt.scatter(x + jitter + offset, y, alpha=0.1, s=5.0,
                            marker=marker, color=color)

    for group_label in ['control', 'temgesic']:

        if group_label == 'temgesic':

            color = 'tab:orange'

            mean_y = []
            unique_run = np.unique(haemo_delays['run'])

            for run in unique_run:
                filter_run = ((haemo_delays['group'] == 'temgesic') &
                              (haemo_delays['run'] == run))
                mean_y.append(np.mean(haemo_delays[filter_run]['a_hat']))

            plt.scatter(unique_run, mean_y, marker='*', s=4.0, alpha=0.9,
                        color=color)
            plt.plot(unique_run, mean_y, color=color, alpha=0.9, lw=1.0)

        else:

            color = 'tab:blue'

            mean_y = []
            unique_run = np.unique(haemo_delays['run'])

            for run in unique_run:
                filter_run = ((haemo_delays['group'] == 'control') &
                              (haemo_delays['run'] == run))
                mean_y.append(np.mean(haemo_delays[filter_run]['a_hat']))

            plt.scatter(unique_run, mean_y, marker='*', s=4.0, alpha=0.9,
                        color=color)
            plt.plot(unique_run, mean_y, color=color, alpha=0.9, lw=1.0)

    plt.xticks(unique_run, [f'Run-{i}' for i in unique_run],
               fontdict=dict(fontsize=12))
    plt.tick_params(axis="y", labelsize=12)
    plt.ylim(0.2 - 0.1, 2.0 + 0.1)

    plt.xlabel("")
    plt.ylabel(r"$\delta$", fontsize=12, rotation=0)

    plt.grid()

    plt.tight_layout()

    suffix = "_shared_maps" if shared_maps else ""
    filename = f"boxplot_a_hat_whole_brain{suffix}"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)
    subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    # Plotting evolution
    fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    for sub in np.unique(haemo_delays['sub']):

        filter_sub = haemo_delays['sub'] == sub

        group = np.unique(haemo_delays[filter_sub]['group'])[0]
        runs = np.sort(np.unique(haemo_delays[filter_sub]['run']))

        runs_short_tr, runs_long_tr = [], []
        mean_a_hat, mean_a_hat_short_tr, mean_a_hat_long_tr = [], [], []
        for run in runs:
            filter_sub_and_run = filter_sub & (haemo_delays['run'] == run)

            t_r = np.unique(haemo_delays[filter_sub_and_run]['t_r'])[0]
            mean_a_hat_ = np.mean(haemo_delays[filter_sub_and_run]['a_hat'])

            mean_a_hat.append(mean_a_hat_)

            if t_r == 2.:
                runs_long_tr.append(run)
                mean_a_hat_long_tr.append(mean_a_hat_)

            elif t_r == .8:
                runs_short_tr.append(run)
                mean_a_hat_short_tr.append(mean_a_hat_)

        ax.plot(runs, mean_a_hat, color=group_color[group], linewidth=1.5,
                alpha=.5)

        ax.scatter(x=runs_long_tr, y=mean_a_hat_long_tr,
                   color=group_color[group], marker='*', alpha=.5)

        ax.scatter(x=runs_short_tr, y=mean_a_hat_short_tr,
                   color=group_color[group], marker='o', alpha=.5)

        last_run_idx = np.argmax(runs)
        ax.text(runs[last_run_idx] + 0.2, mean_a_hat[last_run_idx],
                sub_tag_convert[sub], fontsize=8, color=group_color[group])

    ax.set_xticks(unique_run)
    run_name = [f"Run-{i}" for i in unique_run]
    ax.set_xticklabels(run_name, fontdict=dict(fontsize=15))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\delta$", fontsize=22, rotation=0)
    ax.set_ylim(0.2 - 0.1, 2.0 + 0.1)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.grid()
    plt.tight_layout()

    suffix = "_shared_maps" if shared_maps else ""
    filename = f"evolution_a_hat_whole_brain{suffix}"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)
    subprocess.call(f"pdfcrop {filepath} {filepath}", shell=True)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
