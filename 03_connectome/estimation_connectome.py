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
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


# Global variables
cache_dir = "__cache__"


# Main
if __name__ == '__main__':

    # python3 estimation_connectome.py --preproc-dir preproc_dir --results-dir results_connectome --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--preproc-dir', type=str, default='preproc_data',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    # get all hyper-band fMRI data available
    preproc_dir = os.path.abspath(os.path.normpath(args.preproc_dir))
    func_paths = glob(f"{preproc_dir}/*.nii.gz")

    # Gather func_names and metadata
    l_n_run, l_sub_tag = [], []
    for func_path in func_paths:
        func_name = os.path.basename(func_path)
        l_n_run.append(int(func_name.split('run-')[1][0]))
        l_sub_tag.append(func_name.split('_')[1])

    # Set connectome computation with Yeo 2011 atlas
    yeo = datasets.fetch_atlas_yeo_2011()
    connectome_measure = ConnectivityMeasure(kind='correlation')
    masker = NiftiLabelsMasker(labels_img=yeo['thick_17'], standardize=True,
                               memory=cache_dir)

    # Compute connectome
    if args.verbose:
        print("Starting connectome computation...")

    t0 = time.time()
    time_series = []
    for i, func_path in enumerate(func_paths, start=1):
        time_series.append(masker.fit_transform(func_path))
        if args.verbose:
            print(f"\rmasker.fit_transform files {i}/{len(func_paths)}...",
                  end='')
    print()

    corr_matrices = connectome_measure.fit_transform(time_series)

    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if args.verbose:
          print(f"Donnectome computation done in {delta_t}")

    if args.verbose:
        print("Starting saving...")

  # Save results
    for corr_matrice, n_run, sub_tag in zip(corr_matrices, l_n_run, l_sub_tag):
        sub_results_dir = os.path.join(args.results_dir, sub_tag)
        if not os.path.isdir(sub_results_dir):
            os.makedirs(sub_results_dir)

        sub_results_run_dir = os.path.join(sub_results_dir, f"run-{n_run}")
        if not os.path.isdir(sub_results_run_dir):
            os.makedirs(sub_results_run_dir)

        corr_path = os.path.join(sub_results_run_dir, 'corr_matrice.npy')
        np.save(corr_path, corr_matrice)

    if args.verbose:
        print(f"Saving done")

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
