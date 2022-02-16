""" Plot the T-test on the connectomes. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import argparse
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from nilearn import plotting, datasets
from hemolearn.atlas import fetch_aal3_vascular_atlas


def mask_from_connectome_matrix(th, conn):
    """ Create the mask for the given connectome matrix. """
    assert isinstance(th, str)

    if th == '100%':
        return np.triu(np.ones_like(conn), 1).astype(bool)

    else:
        th = float(th[:-1]) / 100.
        n = int(th * conn.size)
        th_value = np.sort(conn.flatten())[::-1][n]
        val_mask = conn < th_value
        tri_mask = np.triu(np.ones_like(conn), 1).astype(bool)
        return val_mask | tri_mask


# Main
if __name__ == '__main__':

    # python3 plot_t_test_per_run.py --connectome-dir ../03_connectome/results_connectome/ --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
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

        connectome = np.load(connectome_path)
        square_shape = connectome.shape

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(connectome_paths):02d}] Connectome"
                  f" extraction for '{connectome_path}'", end='')

        connectomes.loc[i] = [sub, run, group, connectome.flatten()]

        runs.append(run)

    print()
    runs = np.unique(runs)

    ###########################################################################
    # Plotting
    nrows, ncols = 1, len(runs)
    fig_mat, axis_mat = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 8, nrows * 8))
    nrows, ncols = len(runs), 1
    fig_brain, axis_brain = plt.subplots(nrows, ncols,
                                         figsize=(ncols * 8, nrows * 4))

    _, atlas = fetch_aal3_vascular_atlas()
    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)

    p = 1
    for ax_mat, ax_brain, run in zip(axis_mat, axis_brain, runs):

        filter = connectomes['run'] == run

        if args.verbose:
            print(f"\r[{p}/{len(run)}] Plotting {run}", end='')

        filter_control = filter & (connectomes['group'] == 'control')
        filter_temgesic = filter & (connectomes['group'] == 'temgesic')

        a_control = list(connectomes[filter_control]['connectome'])
        a_temgesic = list(connectomes[filter_temgesic]['connectome'])

        n_control, n_temgesic = len(a_control), len(a_temgesic)
        n_samples = n_control if n_control < n_temgesic else n_temgesic

        controls = np.r_[a_control][:n_samples]
        temgesics = np.r_[a_temgesic][:n_samples]

        _, pval = stats.ttest_rel(controls, temgesics, axis=0)
        neg_log_pval = - np.log10(pval).reshape(square_shape)

        mask = mask_from_connectome_matrix('100%', neg_log_pval)

        sns.heatmap(neg_log_pval, square=True, vmin=0., mask=mask,
                    vmax=3., ax=ax_mat, cbar_kws={"shrink": .7})
        ax_mat.set_title(run)

        plotting.plot_connectome(neg_log_pval, coordinates, alpha=0.5,
                                 edge_threshold=3., title=run,
                                 node_size=15, axes=ax_brain)

        p += 1

    print()
    fig_mat.tight_layout()

    filename = f"t_test_connectomes_mat_per_run"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    fig_mat.savefig(filepath, dpi=300)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    fig_mat.savefig(filepath, dpi=300)
    fig_mat.tight_layout()

    filename = f"t_test_connectomes_brain_per_run"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    fig_brain.savefig(filepath, dpi=300)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    fig_brain.savefig(filepath, dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
