""" Plot simple seed-based correlation analysis. """
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
from nilearn.glm.first_level import (make_first_level_design_matrix,
                                     FirstLevelModel)
from nilearn.input_data import NiftiSpheresMasker
from nilearn.plotting import plot_stat_map


# Global variables
t_r = 2.0
seed_coords = [(0, 10, 34)]
cache_dir = "__cache__"
valid_session = [1, 2]
group_color = dict(control='tab:blue', temgesic='tab:orange')
participants_fname = '../data/nifti_dir/participants.tsv'
participants_path = os.path.abspath(os.path.normpath(participants_fname))
participants = pd.read_csv(participants_path, sep='\t')
sub_tag_to_dicom_tag = dict(zip(participants['participant_id'],
                                participants['DICOM tag']))
dicom_tag_to_sub_tag = dict(zip(participants['DICOM tag'],
                                participants['participant_id']))
selected_group = ['S00634', 'S00669']


# Global functions
def seed_base_analysis(func_data, seed_coords, radius, t_r, verbose=0):
    """ Seed-based correlation analysis.
    """
    seed_masker = NiftiSpheresMasker(seed_coords, radius=radius, t_r=t_r,
                                     memory=cache_dir, memory_level=1,
                                     verbose=verbose)
    seed_time_series = seed_masker.fit_transform(func_data)

    n_scans = seed_time_series.shape[0]
    frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    design_matrix = make_first_level_design_matrix(frametimes,
                                                   add_regs=seed_time_series,
                                                   add_reg_names=["seed"])

    glm = FirstLevelModel(t_r=t_r)
    glm.fit(run_imgs=func_data, design_matrices=design_matrix)

    contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    z_map = glm.compute_contrast(contrast, output_type='z_score')

    return z_map


def th(x, t, absolute=True):
    """Return threshold level to retain t entries of array x."""
    if isinstance(t, str):
        t = float(t[:-1]) / 100.
    elif isinstance(t, float):
        pass
    else:
        raise ValueError(f"t (={t}) type not understtod: shoud be a float"
                         f" between 0 and1 or a string such as '80%'")
    if absolute:
        return np.sort(np.abs(x.flatten()))[-int(t * len(x.flatten()))]
    else:
        return np.sort(x.flatten())[int(t * len(x.flatten()))]


# Main
if __name__ == '__main__':

    # python3 seed_based_highlight.py --plots-dir plots --splitted-session-dir ../data/splitted_session --radius 20 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--splitted-session-dir', type=str,
                        default='splitted_session',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--radius', type=float, default=10,
                        help='Set the radius seed.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.verbose:
        print(f"Creating '{args.plots_dir}'")

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    results_dir = os.path.abspath(os.path.normpath(args.splitted_session_dir))
    pattern_fname = 'sub-*_run_1_*_session.nii.gz'
    pattern = os.path.join(args.splitted_session_dir, pattern_fname)

    plot_params = dict()
    for path in glob(pattern, recursive=True):
        fname = os.path.basename(path)
        sub_tag, session = fname.split('_')[0], fname.split('_')[3]
        plot_params[(sub_tag, session)] = path

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for keys, path in plot_params.items():
        sub_tag, session = keys
        i = 0 if sub_tag == 'sub-05' else 1
        j = 0 if session == 'first' else 1
        z_map = seed_base_analysis(path, seed_coords, args.radius, t_r)
        th_value = th(z_map.get_fdata(), '50%', absolute=True)
        plot_stat_map(z_map, threshold=th_value, axes=ax[i, j],
                      title=f"{sub_tag} (session-{session})")

    filename = (f"seed_base.pdf")
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
