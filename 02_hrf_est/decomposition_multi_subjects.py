""" Simple example to slrda decompose synchropioid subjects. """
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
import uuid
import itertools
import pickle
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from hemolearn import SLRDA
from hemolearn.utils import fmri_preprocess, sort_atoms_by_explained_variances


# Global variables
t_r = 0.8
cache_dir = "__cache__"

# Global functions
def decompose_single_subject_multiple_params(decomp_params, results_dir,
                                             verbose=False):
    """ Helper functions for parallelization. """
    func_path = decomp_params.pop('func_path')

    func_fname = os.path.basename(func_path)
    n_run = int(func_fname.split('run-')[1][0])
    sub_tag = func_fname.split('_')[1]

    # prepare the results directory
    sub_results_dir = os.path.join(results_dir, sub_tag)
    time.sleep(np.random.rand())  # no racing condition
    if not os.path.isdir(sub_results_dir):
        os.makedirs(sub_results_dir, exist_ok=True)

    sub_results_run_dir = os.path.join(sub_results_dir,f"run-{n_run}")
    time.sleep(np.random.rand())  # no racing condition
    if not os.path.isdir(sub_results_run_dir):
        os.makedirs(sub_results_run_dir, exist_ok=True)

    save_dir = os.path.join(sub_results_run_dir, str(uuid.uuid4()))
    time.sleep(np.random.rand())  # no racing condition
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("Starting SLRDA decomposition...")

    slrda = SLRDA(**decomp_params)

    # Decomposition
    try:
        t0 = time.time()
        slrda.fit(func_path)
        delta_t = time.gmtime(time.time() - t0)
        delta_t = time.strftime("%H h %M min %S s", delta_t)

        if verbose:
            print(f"Decomposition done in {delta_t}")

        # Gather all results
        hrf_ref = slrda.v_init
        masker_kwargs = dict(mask_img=slrda.mask_full_brain, t_r=t_r,
                             memory='.', memory_level=1, verbose=0)
        a_hat, v_hat = slrda.a_hat, slrda.v_hat
        u_hat, z_hat, variances = sort_atoms_by_explained_variances(
            slrda.u_hat, slrda.z_hat, slrda.v_hat, slrda.hrf_rois)
        results = dict(func_path=func_path, hrf_ref=hrf_ref,
                       hrf_rois=slrda.hrf_rois,
                       masker_kwargs=masker_kwargs, a_hat=a_hat,
                       v_hat=v_hat, u_hat=u_hat, z_hat=z_hat,
                       variances=variances)

        # Gather all metadatas
        metadata = dict(sub_tag=sub_tag, func_path=func_path,
                        n_run=n_run, decomp_params=decomp_params)

    except Exception as e:
        print(f"Error appear ('{e}'), skipping decomposition and saving "
              f"empty results")
        results, metadata = dict(), dict()

    if verbose:
        print("Starting saving...")

    t0 = time.time()

    # Send back results
    picklefname = os.path.join(save_dir, 'results.pkl')
    with open(picklefname, 'wb') as pfile:
        pickle.dump(results, pfile)

    # Send back metadata
    jsonfname = os.path.join(save_dir, 'metadata.json')
    with open(jsonfname, 'w') as jsonfile:
        json.dump(metadata, jsonfile)

    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if verbose:
        print(f"Saving done in {delta_t}")


# Main
if __name__ == '__main__':

    # python3 decomposition_multi_subjects.py --max-iter 100 --seed 0 --preproc-dir preproc_dir --results-dir results_slrda --cpu 4 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--preproc-dir', type=str, default='preproc_data',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    # set seed
    seed = np.random.randint(0, 1000) if args.seed is None else args.seed
    print(f'Seed used = {seed}')

    # get all hyper-band fMRI data available
    preproc_dir = os.path.abspath(os.path.normpath(args.preproc_dir))
    func_paths = glob(f"{preproc_dir}/*task-hbrest*.nii.gz")

    # parameters grid to be define
    param_grid = {
        'func_path': func_paths,
        'hrf_atlas': ['basc'],
        'n_scales': ['scale064', 'scale122', 'scale325'],
        'n_atoms': [10, 20, 30],
        'lbda': [0.001, 0.1, 0.9],
        'hrf_model': ['scaled_hrf'],
        'prox_u': ['l1-positive-simplex'],
        'max_iter': [args.max_iter],
        'n_times_atom': [60],
        'eps': [1.0e-3],
        'cache_dir': [cache_dir],
        't_r': [t_r],
        'verbose': [2],
        'random_state':[seed],
    }

    l_params = [dict(zip(param_grid, x))
                for x in itertools.product(*param_grid.values())
                ]

    # main loop
    Parallel(n_jobs=args.cpu, verbose=100)(delayed(
                    decompose_single_subject_multiple_params)(
            decomp_params, results_dir=args.results_dir, verbose=args.verbose)
                            for decomp_params in l_params)

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
