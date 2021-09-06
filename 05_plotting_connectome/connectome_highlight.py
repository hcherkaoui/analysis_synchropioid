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
selected_group = ['S00634', 'S00669']
valid_session = [1, 2]
participants_fname = '../data/nifti_dir/participants.tsv'
participants_path = os.path.abspath(os.path.normpath(participants_fname))
participants = pd.read_csv(participants_path, sep='\t')
sub_tag_convert = dict(zip(participants['participant_id'],
                           participants['DICOM tag']))


# Main
if __name__ == '__main__':

    # python3 connectome_highlight.py --plots-dir plots --results-dir ../03_connectome_splitted_session/results_connectome_splitted_session/ --verbose 1

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

    connectomes = dict()
    for i, results_path in enumerate(results_paths):

        results_fname = os.path.basename(results_path)

        run_dir_path = os.path.dirname(results_path)
        sub_dir_path = os.path.dirname(run_dir_path)

        connectome = np.load(results_path)
        n_run = int(os.path.basename(run_dir_path)[-1])
        sub_tag = os.path.basename(sub_dir_path)
        session = int(results_fname.split('.')[0].split('_')[-1])

        filter_sub = sub_tag_convert[sub_tag] in selected_group
        filter_run = n_run == 1
        if filter_sub & filter_run:
            connectomes[(sub_tag_convert[sub_tag], session)] = connectome

    yeo = datasets.fetch_atlas_yeo_2011()

    # Plotting evolution
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, sub_tag in enumerate(selected_group):
        for j, session in enumerate(valid_session):
            conn = connectomes[(sub_tag, session)]
            coor = find_parcellation_cut_coords(labels_img=yeo['thick_17'])
            plot_connectome(conn, coor, edge_threshold="80%",
                            axes=ax[i, j], display_mode='z',
                            title=f"{sub_tag} (session-{session})")

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
