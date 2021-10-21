#!/usr/bin/env python3
""" Simple example to define a mask from the PET data. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting, image


import warnings
warnings.filterwarnings("ignore")


def threshold(pet_fname, th='10%'):
    th = float(th[:-1]) / 100.
    pet_raw = image.get_data(image.load_img(pet_fname)).ravel()
    return th * np.nanmax(pet_raw)


if __name__ == '__main__':

    # ./binarized_pet -v --show-plot --pet-filename ../data/pet_dir/con_0001.nii --list-threshold 90% --cluster-threshold 0 --mask-filename pet_mask.nii.gz

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--pet-filename', type=str,
                        help='PET data filename.')
    parser.add_argument('-o', '--mask-filename', type=str,
                        default='pet_mask.nii.gz',
                        help='Output mask filename.')
    parser.add_argument('-t', '--list-threshold', type=str, nargs='+',
                        default=['90%', '95%'],
                        help='List of threshold percent.')
    parser.add_argument('-c', '--cluster-threshold', type=int, default=150,
                        help='Minimum voxels per cluster.')
    parser.add_argument('-s', '--show-plot',
                        help='Option to display the PET-mask',
                        action='store_true',
                        default=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true',
                        default=False)
    args = parser.parse_args()

    n_cut_coor = 5
    cut_coor = np.linspace(-10, 10, n_cut_coor)
    figsize = (2 * n_cut_coor, 3 * len(args.list_threshold))

    _, axis = plt.subplots(len(args.list_threshold), 1, figsize=figsize)
    for i, th in enumerate(args.list_threshold):
        th_value = threshold(args.pet_filename, th)
        pet_clusters = image.threshold_img(
                            img=args.pet_filename, threshold=th_value,
                            cluster_threshold=args.cluster_threshold
                            )
        pet_mask = image.binarize_img(pet_clusters)

        path, fname = os.path.split(args.mask_filename)
        fname = f"th-{th}_" + fname
        fpath = os.path.join(path, fname)

        if args.verbose:
            print(f"PET-mask saved under '{args.mask_filename}'.")

        pet_mask.to_filename(fpath)

        axes = axis[i] if isinstance(axis, list) else axis
        plotting.plot_stat_map(pet_mask, title=f"th={th}", axes=axes,
                               display_mode='z', cmap='black_blue_r',
                               cut_coords=np.linspace(-10, 10, 5))

    if args.show_plot:
        plt.show()
