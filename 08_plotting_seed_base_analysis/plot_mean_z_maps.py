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
import numpy as np
from nilearn import input_data, plotting, image


# set fontsize
sns.set(font_scale=0.7)


# Main
if __name__ == '__main__':

    # python3 plot_mean_z_maps.py --z-maps-dir ../05_seed_base_analysis/z_maps/ --plots-dir plots --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--z-maps-dir', type=str, default='z_maps_dir',
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
                                         f"sub-*_run-*_group-*_tr-*_z_map"
                                         f".nii.gz")
    z_maps_paths = glob(template_z_maps_paths)

    connectivity = pd.DataFrame(columns=['sub', 'run', 'group', 'norm'])
    for i, z_maps_path in enumerate(z_maps_paths):

        z_maps_path = os.path.normpath(z_maps_path)
        z_maps_name = os.path.basename(z_maps_path)
        sub, run, group, _, _, _ = z_maps_name.split('_')

        sub = int(sub.split('-')[-1])
        run = f"Run-{int(run.split('-')[-1])}"
        group = group.split('-')[-1]

        z_maps = image.load_img(z_maps_path).get_fdata().flatten()
        conn_z_maps = np.mean(z_maps)

        if args.verbose:
            print(f"\r[{i+1:02d}/{len(z_maps_paths):02d}] Connectivity "
                  f"computation for '{z_maps_path}'", end='')

        connectivity.loc[i] = [sub, run, group, conn_z_maps]

    ###########################################################################
    # Plotting
    palette = {"control":'tab:blue',
               "temgesic":'tab:orange'}
    order = ["Run-1", "Run-2", "Run-3", "Run-4", "Run-5"]

    plt.figure('Boxplot', figsize=(3, 2))
    ax = sns.violinplot(data=connectivity, x='run', y='norm', hue='group',
                        split=True, palette=palette, order=order,
                        linewidth=0.75, width=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.tight_layout()
    filename = os.path.join(args.plots_dir, f"connectivity_boxplot")

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
