""" Plot the T-test on vascular maps. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
import subprocess
import argparse
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from nilearn import input_data, plotting
from hemolearn import atlas


# Main
if __name__ == '__main__':

    # python3 t_test_per_run.py --z-maps-dir output_dir --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--z-maps-dir', type=str, default='output_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Plots directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the seed base analysis z-maps
    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir, exist_ok=True)

    template_z_maps_paths = os.path.join(args.z_maps_dir,
                                                f"sub-*_run-*_group-*_"
                                                f"tr-*_*.nii.gz")
    z_maps_paths = glob(template_z_maps_paths)

    mask, _ = atlas.fetch_aal3_vascular_atlas()
    masker = input_data.NiftiMasker(mask_img=mask, smoothing_fwhm=18.)
    masker.fit(z_maps_paths)

    runs = []
    columns = ['sub', 'run', 'group', 'z_map']
    z_maps = pd.DataFrame(columns=columns)
    for i, z_maps_path in enumerate(z_maps_paths):

        z_maps_path = os.path.normpath(z_maps_path)
        z_maps_name = os.path.basename(z_maps_path)
        chunks = z_maps_name.split('_')
        sub, run, group = chunks[0], chunks[1], chunks[2]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1]

        z_map = masker.transform_single_imgs(z_maps_path)

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(z_maps_paths):02d}] Z-"
                  f"maps extraction for '{z_maps_path}'", end='')

        z_maps.loc[i] = [sub, run, group, z_map.flatten()]

        runs.append(run)

    runs = np.unique(runs)

    ###########################################################################
    # Plotting
    nrows, ncols = len(runs), 1
    _, axis = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 2))

    for ax, run in zip(axis, runs):

        filter = z_maps['run'] == run

        filter_control = filter & (z_maps['group'] == 'control')
        filter_temgesic = filter & (z_maps['group'] == 'temgesic')

        a_control = list(z_maps[filter_control]['z_map'])
        a_temgesic = list(z_maps[filter_temgesic]['z_map'])

        n_control, n_temgesic = len(a_control), len(a_temgesic)
        n_samples = n_control if n_control < n_temgesic else n_temgesic

        controls = np.r_[a_control][:n_samples]
        temgesics = np.r_[a_temgesic][:n_samples]

        _, pval = stats.ttest_rel(controls, temgesics, axis=0)

        try:

            neg_log_pval_img = masker.inverse_transform(-np.log10(pval))

            plotting.plot_stat_map(neg_log_pval_img, title=run,
                                   display_mode='z',
                                   cut_coords=np.linspace(-30, 60, 8),
                                   axes=ax, threshold=1., vmax=5.)

        except TypeError:

            continue

    filename = f"t_test_z_maps_per_run"

    filepath = os.path.join(args.plots_dir, filename + '.pdf')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)
    subprocess.call(f"pdfcrop {filepath} {filepath}",
                    shell=True)

    filepath = os.path.join(args.plots_dir, filename + '.png')
    print(f"Saving plot at '{filepath}'")
    plt.savefig(filepath, dpi=300)


    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
