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


# Main
if __name__ == '__main__':

    # python3 plot_t_test_per_run.py --vascular-maps-dir output_dir --plots-dir plots --verbose 1
    # python3 plot_t_test_per_run.py --vascular-maps-dir output_shared_maps_dir --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--vascular-maps-dir', type=str, default='output_dir',
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

    template_vascular_maps_paths = os.path.join(args.vascular_maps_dir,
                                                f"sub-*_run-*_group-*_"
                                                f"tr-*_*.nii.gz")
    vascular_maps_paths = glob(template_vascular_maps_paths)

    masker = input_data.NiftiMasker()
    masker.fit(vascular_maps_paths)

    runs = []
    columns = ['sub', 'run', 'group', 'vascular_map']
    vascular_maps = pd.DataFrame(columns=columns)
    for i, vascular_maps_path in enumerate(vascular_maps_paths):

        vascular_maps_path = os.path.normpath(vascular_maps_path)
        vascular_maps_name = os.path.basename(vascular_maps_path)
        chunks = vascular_maps_name.split('_')
        sub, run, group, t_r = chunks[0], chunks[1], chunks[2], chunks[3]

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1]
        t_r = float(t_r.split('-')[-1])

        vascular_map = masker.transform_single_imgs(vascular_maps_path)

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(vascular_maps_paths):02d}] Vascular "
                  f"maps extraction for '{vascular_maps_path}'", end='')

        vascular_maps.loc[i] = [sub, run, group, vascular_map.flatten()]

        runs.append(run)

    runs = np.sort(np.unique(runs))

    ###########################################################################
    # Plotting
    nrows, ncols = len(runs), 1
    _, axis = plt.subplots(nrows, ncols, figsize=(ncols * 20, nrows * 2))

    for ax, run in zip(axis, runs):

        filter = vascular_maps['run'] == run

        filter_control = filter & (vascular_maps['group'] == 'control')
        filter_temgesic = filter & (vascular_maps['group'] == 'temgesic')

        a_control = list(vascular_maps[filter_control]['vascular_map'])
        a_temgesic = list(vascular_maps[filter_temgesic]['vascular_map'])

        n_control, n_temgesic = len(a_control), len(a_temgesic)
        n_samples = n_control if n_control < n_temgesic else n_temgesic

        controls = np.r_[a_control][:n_samples]
        temgesics = np.r_[a_temgesic][:n_samples]

        _, pval = stats.ttest_rel(controls, temgesics, axis=0)

        try:

            neg_log_pval_img = masker.inverse_transform(-np.log10(pval))

            colorbar = True if run == 'Run-5' else False

            plotting.plot_stat_map(neg_log_pval_img, title=run, cmap='bwr',
                                   display_mode='z', colorbar=colorbar,
                                   cut_coords=np.linspace(-40, 70, 15),
                                   axes=ax, threshold=1.0, vmax=3.0)

        except TypeError:

            continue

    filename = f"t_test_vascular_maps_per_run"
    filename += f"_from_{args.vascular_maps_dir}"

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
