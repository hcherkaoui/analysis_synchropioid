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
from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords, plot_connectome


# Global variables
participants_fname = '../data/nifti_dir/participants.tsv'
participants_path = os.path.abspath(os.path.normpath(participants_fname))
participants = pd.read_csv(participants_path, sep='\t')

sub_tag_to_dicom_tag = dict(zip(participants['participant_id'],
                                participants['DICOM tag']))
dicom_tag_to_sub_tag = dict(zip(participants['DICOM tag'],
                                participants['participant_id']))

selected_keys = {('S00634', 'run-1', 1): (0, 0),
                 ('S00634', 'run-1', 2): (0, 1),
                 ('S00669', 'run-1', 1): (1, 0),
                 ('S00669', 'run-3', 1): (1, 1),
                }


# Main
if __name__ == '__main__':

    # python3 connectome_highlight.py --plots-dir plots --splitted-session-dir ../03_connectome_splitted_session/results_connectome_splitted_session/ --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--splitted-session-dir', type=str,
                        default='results_connectome_splitted_session',
                        help='Set the name of the results directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.verbose:
        print(f"Saving figures under '{args.plots_dir}' directory")

    results_dir = os.path.abspath(os.path.normpath(args.splitted_session_dir))
    pattern_fname = '**/corr_matrice_session*.npy'
    pattern = os.path.join(args.splitted_session_dir, pattern_fname)
    results_paths = list(glob(pattern, recursive=True))

    if len(results_paths) == 0:
        raise ValueError(f"No results found under "
                         f"'{args.splitted_session_dir}'")

    connectomes = dict()
    sub_tags, runs, sessions = [], [], []
    for path in results_paths:
        sub_tag = path.split('/')[-3]
        n_run = path.split('/')[-2]
        fname = path.split('/')[-1]
        n_session = int(fname.split('-')[-1].split('.')[0])

        filter = ((sub_tag_to_dicom_tag[sub_tag], n_run, n_session)
                  in list(selected_keys.keys()))
        if filter:
            sub_tags.append(sub_tag_to_dicom_tag[sub_tag])
            runs.append(n_run)
            sessions.append(n_session)

            key = (sub_tag_to_dicom_tag[sub_tag], n_run, n_session)
            connectomes[key] = np.load(path)

    yeo = datasets.fetch_atlas_yeo_2011()
    coor = find_parcellation_cut_coords(labels_img=yeo['thick_17'])

    # Plotting evolution
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for keys, axis in selected_keys.items():
        sub_tag, n_run, n_session = keys
        conn = connectomes[(sub_tag, n_run, n_session)]
        plot_connectome(conn[0], coor, edge_threshold="80%",
                        axes=ax[axis], display_mode='z',
                        title=(f"{sub_tag}|{n_run}|session-{n_session}"))

    filename = (f"highlight_connectome.pdf")
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
