""" Simple example to compute connectome on synchropioid subjects. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import gc
import time
from glob import glob
import argparse
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn import  plotting, connectome, input_data, datasets, image
from hemolearn.atlas import fetch_aal3_vascular_atlas


# Define group label
temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


def transform_img(atlas, cache_dir, func_path, confounds_path):
    """ Extract the time series. """

    func_name = os.path.basename(func_path)
    task_type = func_name.split('task-')[1].split('_')[0]
    t_r = 0.8 if task_type == 'hbrest' else 2.0

    masker = input_data.NiftiLabelsMasker(labels_img=atlas,
                                          standardize=True,
                                          t_r=t_r, smoothing_fwhm=6.,
                                          detrend=True, low_pass=0.1,
                                          high_pass=0.01, memory=cache_dir,
                                          memory_level=1)
    # masker = input_data.NiftiMapsMasker(maps_img=atlas, standardize=True,
    #                                     t_r=t_r, smoothing_fwhm=12.,
    #                                     detrend=True, low_pass=0.1,
    #                                     high_pass=0.01, memory=cache_dir,
    #                                     memory_level=1)

    time_to_discard = 30  # seconds
    index = slice(int(time_to_discard / t_r), None)
    func_img = image.index_img(func_path, index)
    confounds = pd.read_csv(confounds_path, sep='\t')[index]

    time_series = masker.fit_transform(func_img, confounds=confounds)

    del masker
    gc.collect()

    return time_series


# Main
if __name__ == '__main__':

    # python3 estimation_connectome.py --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir/ --results-dir results_connectome --plots-dir plots --cpu 3 --verbose 10

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=str, default='z_maps',
                        help='Set the name of the results directory.')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Set the name of the plots directory.')
    parser.add_argument('--bids-root-dir', type=str,
                        default='fmri_nifti_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--cache-dir', type=str,
                        default='__cache__',
                        help='Set the name of the cache directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Collect the functional data
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if args.verbose:
        print(f"Saving connectomes under '{args.results_dir}' directory")

    if args.verbose:
        print(f"Saving connectomes plots under '{args.plots_dir}' directory")

    template_func_paths = os.path.join(args.bids_root_dir,
                                       f"derivatives/sub-*/func/"
                                       f"sub-*_task-rest_run-*_space-"
                                       f"MNI152Lin_desc-preproc_bold.nii.gz")
    func_paths = glob(template_func_paths)

    participants_fname = os.path.join(args.bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                            participants['DICOM tag']))

    ###########################################################################
    # Gather func_names and metadata
    l_run, l_sub, l_group, l_t_r, confounds_paths = [], [], [], [], []
    for func_path in func_paths:

        func_name = os.path.basename(func_path)
        dir_name = os.path.dirname(func_path)

        run = int(func_name.split('run-')[1][0])
        sub = func_name.split('_')[0]
        task = func_name.split('task-')[-1].split('_')[0]
        confounds_path = os.path.join(dir_name,
                                      f"{sub}_task-{task}_run-{run}_desc"
                                      f"-confounds_timeseries.tsv")

        group = ('temgesic' if sub_tag_convert[sub] in temgesic_group
                 else 'control')
        task_type = func_name.split('task-')[1].split('_')[0]
        t_r = 0.8 if task_type == 'hbrest' else 2.0

        l_run.append(run)
        l_sub.append(sub)
        l_group.append(group)
        l_t_r.append(t_r)
        confounds_paths.append(confounds_path)

    ###########################################################################
    # Compute connectome
    if args.verbose:
        print("Starting connectome computation...")

    t0 = time.time()

    # atlas = datasets.fetch_atlas_msdl()['maps']
    _, atlas = fetch_aal3_vascular_atlas()
    # atlas = datasets.fetch_atlas_yeo_2011()['thick_17']

    time_series = Parallel(n_jobs=args.cpu, verbose=args.verbose)(
    delayed(transform_img)(atlas, args.cache_dir, func_path, confounds_path)
            for func_path, confounds_path in zip(func_paths, confounds_paths))

    connectome_measure = connectome.ConnectivityMeasure(kind='correlation')
    corr_matrices = connectome_measure.fit_transform(time_series)

    del connectome_measure
    gc.collect()

    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if args.verbose:
          print(f"Connectome computation done in {delta_t}")

    ###########################################################################
    # Save results
    iterates = zip(corr_matrices, l_run, l_sub, l_group, l_t_r)
    for corr_matrice, run, sub, group, t_r in iterates:

        filename = (f"corr_matrice_{sub}_run-{run}_tr-{t_r}_"
                    f"group-{group}")

        filepath = os.path.join(args.results_dir, filename + '.npy')
        np.save(filepath, corr_matrice)

        _, ax = plt.subplots(figsize=(4,4))
        plotting.plot_matrix(corr_matrice, tri='lower', colorbar=True, axes=ax)
        plt.tight_layout()
        filepath = os.path.join(args.plots_dir, filename + '.png')
        plt.savefig(filepath, dpi=300)

    if args.verbose:
        print(f"Matrices saved done")

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
