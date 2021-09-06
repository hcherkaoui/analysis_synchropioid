""" Simple example to Nilearn-preprocess synchropioid subjects. """
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
import pandas as pd
from joblib import Parallel, delayed
from hemolearn.utils import fmri_preprocess


# Global variables
cache_dir = "__cache__"

# Global functions
def nilearn_preprocessing(func_path, preproc_dir, use_t1_mask, use_confounds,
                          verbose):
    """ Nilearn preprocessing of the Synchropioid fMRI data.
    """
    t_r = 0.8 if 'hbrest' in func_path else 2.0

    func_path = os.path.normpath(func_path)
    func_dir = os.path.dirname(func_path)
    func_fname = os.path.basename(func_path)
    func_tag = func_fname.split('space')[0]

    # confounds filename
    confounds_fname = f'{func_tag}desc-confounds_timeseries.tsv'
    confounds_path = os.path.join(func_dir, confounds_fname)

    # mask filename
    if use_t1_mask:
        mask_fname = (f'{func_tag}space-MNI152Lin_desc-brain_mask.nii.gz')
        mask_path = os.path.join(func_dir, mask_fname)
    else:
        mask_path = None

    # clean confounds filename
    if use_confounds:
        clean_confounds_fname = 'clean_'
        clean_confounds_fname += os.path.basename(confounds_fname)
        clean_confounds_path = os.path.join(preproc_dir, clean_confounds_fname)
        confounds = pd.read_csv(confounds_path, sep='\t')
        for colname in confounds.columns:
            confounds[colname].fillna((confounds[colname].mean()),
                                      inplace=True)
        confounds.to_csv(clean_confounds_path, index=False, sep="\t")
    else:
        clean_confounds_path = None

    # preproc filename
    preproc_fname = 'niprepro_' + os.path.basename(func_fname)
    preproc_path = os.path.join(preproc_dir, preproc_fname)

    preproc_params = dict(func_fname=func_path, mask_img=mask_path,
                          sessions=None, smoothing_fwhm=6.0,
                          standardize=True, detrend=True,
                          low_pass=0.1, high_pass=0.01, t_r=t_r,
                          memory_level=1, memory=cache_dir,
                          verbose=0, confounds=clean_confounds_path,
                          preproc_fname=preproc_path)

    if verbose:
        print(f"Starting preprocessing of {func_fname}...")

    t0 = time.time()
    fmri_preprocess(**preproc_params)
    delta_t = time.gmtime(time.time() - t0)
    delta_t = time.strftime("%H h %M min %S s", delta_t)

    if verbose:
        print(f"Preprocessing of {func_fname} done in {delta_t}")


# Main
if __name__ == '__main__':

    # python3 nilearn_preprocessing.py -m -i nifti_dir/derivatives/ -o preproc_dir --cpu 4 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i','--nifti-dir', default='nifti_dir/derivatives/',
                        help='Name of the Nifti preproc directory')
    parser.add_argument('-o','--preproc-dir', default='preproc_dir',
                        help='Name of the Nifti Nilearn preproc directory')
    parser.add_argument('-m', '--use-t1mask',
                        help='Option to use T1-masking to preprocess the data',
                        action='store_true', default=False)
    parser.add_argument('-c', '--use-confounds',
                        help='Option to use confounds to preprocess the data',
                        action='store_true', default=False)
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isdir(args.preproc_dir):
        os.makedirs(args.preproc_dir)

    # get all fMRI files
    glob_pattern = (f'{args.nifti_dir}/sub-*/func/sub-*_task-*'
                    f'_run-*_space-*_desc-preproc_bold.nii.gz')
    func_paths = glob(glob_pattern)

    # main loop
    Parallel(n_jobs=args.cpu, verbose=100)(delayed(
                    nilearn_preprocessing)(
                        func_path, args.preproc_dir, args.use_t1mask,
                        args.use_confounds, args.verbose)
                            for func_path in func_paths)

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
