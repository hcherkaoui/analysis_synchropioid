""" Simple example to BDA decompose synchropioid subjects. """
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
from hemolearn import BDA


# define group label
temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


# Global functions
def decompose_single_subject_multiple_params(decomp_params, results_dir,
                                             sub_tag_convert, verbose=False):
    """ Helper functions for parallelization. """
    # func_path should be passed to 'fit' method
    func_path = decomp_params.pop('func_path')

    # defined the confound path
    confound_path = func_path.split('space')[0]
    confound_path += 'desc-confounds_timeseries.tsv'

    # fetch 't_r' from func_path
    task_type = os.path.basename(func_path).split('task-')[1].split('_')[0]
    t_r = 0.8 if task_type == 'hbrest' else 2.0
    decomp_params['t_r'] = t_r

    # fetch metadata
    func_fname = os.path.basename(func_path)
    n_run = int(func_fname.split('run-')[1][0])
    sub_tag = func_fname.split('_')[0]
    group_label = ('temgesic' if sub_tag_convert[sub_tag] in temgesic_group
                              else 'control')

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
        print("Starting BDA decomposition...")

    bda = BDA(**decomp_params)

    # Decomposition
    try:
        t0 = time.time()
        bda.fit(func_path, confound_fnames=confound_path)
        delta_t = time.gmtime(time.time() - t0)
        delta_t = time.strftime("%H h %M min %S s", delta_t)

        if verbose:
            print(f"Decomposition done in {delta_t}")

        # Gather all results
        masker_kwargs = dict(mask_img=bda.mask_full_brain, t_r=t_r,
                             memory_level=1, verbose=0)
        results = dict(hrf_rois=bda.hrf_rois, masker_kwargs=masker_kwargs,
                       a_hat=bda.a_hat, a_hat_img=bda.a_hat_img,
                       v_hat=bda.v_hat, u_hat=bda.u_hat,
                       u_hat_img=bda.u_hat_img, z_hat=bda.z_hat)

        # Gather all metadatas
        metadata = dict(sub_tag=sub_tag, func_path=func_path, t_r=t_r,
                        n_run=n_run, group_label=group_label,
                        decomp_params=decomp_params)

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

    # python3 decomposition_multi_subjects.py --max-iter 200 --seed 0 --bids-root-dir fmri_nifti_dir/ --results-dir results_hrf_estimation --cpu 60 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Max number of iterations to train the '
                        'learnable networks.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--bids-root-dir', type=str, default='bids_root_data',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--results-dir', type=str, default='results_dir',
                        help='Set the name of the results directory.')
    parser.add_argument('--cache-dir', type=str, default='__cache__',
                        help='Caching directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Launch the decomposition
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    # set seed
    seed = np.random.randint(0, 1000) if args.seed is None else args.seed
    print(f'Seed used = {seed}')

    # get all hyper-band fMRI data available
    bids_root_dir = os.path.abspath(os.path.normpath(args.bids_root_dir))

    participants_fname = os.path.join(bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                               participants['DICOM tag']))

    template_func_paths = os.path.join(args.bids_root_dir,
                                       f"derivatives/sub-*/func/"
                                       f"sub-*_task-*rest_run-*_space-"
                                       f"MNI152Lin_desc-preproc_bold.nii.gz")
    func_paths = glob(template_func_paths)

    # parameters grid to be define
    param_grid = {
        'func_path': func_paths,
        'hrf_atlas': ['aal3'],  # tuned
        'n_atoms': [20],  # tuned
        'shared_spatial_maps': [False],
        'lbda': list(np.logspace(-2, -1, 8)),  # tuned
        'hrf_model': ['scaled_hrf'],
        'prox_u': ['l1-positive-simplex'],
        'standardize': [True],
        'detrend': [True],
        'low_pass': [0.1],
        'high_pass': [0.01],
        'max_iter': [args.max_iter],
        'n_times_atom': [60],
        'eps': [1.0e-4],
        'cache_dir': [args.cache_dir],
        'verbose': [2],
        'random_state':[seed],
    }

    l_params = [dict(zip(param_grid, x))
                for x in itertools.product(*param_grid.values())
                ]

    # main loop
    Parallel(n_jobs=args.cpu, verbose=100)(delayed(
                    decompose_single_subject_multiple_params)(
            decomp_params, results_dir=args.results_dir,
            sub_tag_convert=sub_tag_convert, verbose=args.verbose)
                            for decomp_params in l_params)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
