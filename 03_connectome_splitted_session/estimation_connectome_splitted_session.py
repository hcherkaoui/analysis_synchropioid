""" Simple example to compute connectome on synchropioid subjects. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import time
from glob import glob
import argparse
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


# Global variables
cache_dir = "__cache__"
selected_group = ['S00634', 'S00669']
participants_fname = '../data/nifti_dir/participants.tsv'
participants_path = os.path.abspath(os.path.normpath(participants_fname))
participants = pd.read_csv(participants_path, sep='\t')
sub_tag_to_dicom_tag = dict(zip(participants['participant_id'],
                                participants['DICOM tag']))
dicom_tag_to_sub_tag = dict(zip(participants['DICOM tag'],
                                participants['participant_id']))


# Main
if __name__ == '__main__':

    # python3 estimation_connectome_splitted_session.py --preproc-dir preproc_dir --results-dir results_connectome_splitted_session --splitted_session-dir ../data/splitted_session --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--preproc-dir', type=str, default='preproc_data',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--tmp-dir', type=str, default='tmp',
                        help='Set the name of the temporary directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.isdir(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    # get all hyper-band fMRI data available
    preproc_dir = os.path.abspath(os.path.normpath(args.preproc_dir))
    func_paths = glob(f"{preproc_dir}/*.nii.gz")

    if args.verbose:
        print("Gathering data...")

    t0 = time.time()

    # Gather func_names and metadata
    len_session = int(5 * 60 / 2)
    len_to_discard = 10
    l_n_run, l_sub_tag = [], []
    l_func_path_first_session, l_func_path_second_session = [], []
    for func_path in func_paths:
        func_name = os.path.basename(func_path)

        n_run = int(func_name.split('run-')[1][0])
        sub_tag = func_name.split('_')[1]

        filter_run = n_run == 1
        filter_sub = sub_tag in [dicom_tag_to_sub_tag[tag]
                                 for tag in selected_group]

        if not (filter_run & filter_sub):
            continue

        img = image.load_img(func_path)
        n_scans = img.shape[-1]

        idx_first_session = range(len_to_discard, len_session + len_to_discard)
        img_first_session = image.index_img(img, idx_first_session)
        filename = f'{sub_tag}_run_1_first_session.nii.gz'
        first_session_path = os.path.join(args.tmp_dir, filename)
        img_first_session.to_filename(first_session_path)

        idx_second_session = range(n_scans - len_session, n_scans)
        img_second_session = image.index_img(img, idx_second_session)
        filename = f'{sub_tag}_run_1_second_session.nii.gz'
        second_session_path = os.path.join(args.tmp_dir, filename)
        img_second_session.to_filename(second_session_path)

        l_n_run.append(n_run)
        l_sub_tag.append(sub_tag)
        l_func_path_first_session.append(first_session_path)
        l_func_path_second_session.append(second_session_path)

    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if args.verbose:
          print(f"Gathering done in {delta_t}")

    # Set connectome computation with Yeo 2011 atlas
    yeo = datasets.fetch_atlas_yeo_2011()
    connectome_measure_session_1 = ConnectivityMeasure(kind='correlation')
    connectome_measure_session_2 = ConnectivityMeasure(kind='correlation')
    masker_session_1 = NiftiLabelsMasker(labels_img=yeo['thick_17'],
                                         standardize=True, memory=cache_dir)
    masker_session_2 = NiftiLabelsMasker(labels_img=yeo['thick_17'],
                                         standardize=True, memory=cache_dir)
    # Compute connectome
    if args.verbose:
        print("Starting connectome computation...")

    t0 = time.time()
    time_series_session_1, time_series_session_2 = [], []
    iterations = zip(l_func_path_first_session, l_func_path_second_session)
    for i, func_paths in enumerate(iterations, start=1):
        func_path_first_session, func_path_second_session = func_paths

        time_series = masker_session_1.fit_transform(func_path_first_session)
        time_series_session_1.append(time_series)
        time_series = masker_session_2.fit_transform(func_path_second_session)
        time_series_session_2.append(time_series)
        if args.verbose:
            print(f"\rmasker.fit_transform files {i}"
                  f"/{len(l_func_path_first_session)}...", end='')
    print()

    corr_matrices_session_1 = \
            connectome_measure_session_1.fit_transform(time_series_session_1)
    corr_matrices_session_2 = \
            connectome_measure_session_2.fit_transform(time_series_session_2)

    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if args.verbose:
          print(f"Donnectome computation done in {delta_t}")

    if args.verbose:
        print("Starting saving...")

  # Save results
    iterations = zip(corr_matrices_session_1,
                     corr_matrices_session_2,
                     l_n_run, l_sub_tag)
    for corr_mat_1, corr_mat_2, n_run, sub_tag in iterations:
        sub_results_dir = os.path.join(args.results_dir, sub_tag)
        if not os.path.isdir(sub_results_dir):
            os.makedirs(sub_results_dir)

        sub_results_run_dir = os.path.join(sub_results_dir, f"run-{n_run}")
        if not os.path.isdir(sub_results_run_dir):
            os.makedirs(sub_results_run_dir)

        corr_path = os.path.join(sub_results_run_dir,
                                 'corr_matrice_session_1.npy')
        np.save(corr_path, corr_mat_1)

        corr_path = os.path.join(sub_results_run_dir,
                                 'corr_matrice_session_2.npy')
        np.save(corr_path, corr_mat_2)

    if args.verbose:
        print(f"Saving done")

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
