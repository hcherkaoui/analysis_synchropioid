""" Seed base analysis from the PET seed. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import time
from glob import glob
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from nilearn.glm.first_level import (make_first_level_design_matrix,
                                     FirstLevelModel)
from nilearn.input_data import NiftiMasker


# Define group label
temgesic_group = ['S00634_1558', 'S00651_1695', 'S00669_1795', 'S00726_2083',
                  'S00748_2205', 'S00791_2518', 'S00805_2631']


# Global functions
def seed_base_analysis(func_data, mask_img, t_r, cache_dir, verbose=0):
    """ Seed-based correlation analysis.
    """
    seed_masker = NiftiMasker(mask_img=mask_img, standardize=True, t_r=t_r,
                              memory_level=1, memory=cache_dir,
                              verbose=verbose)


    mean_seed_time_serie = seed_masker.fit_transform(func_data).mean(axis=1)

    n_scans = mean_seed_time_serie.shape[0]
    frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)

    design_matrix = make_first_level_design_matrix(
                                    frame_times=frametimes,
                                    add_regs=mean_seed_time_serie[:, None],
                                    add_reg_names=["seed"])

    glm = FirstLevelModel(t_r=t_r, standardize=True)
    glm.fit(run_imgs=func_data, design_matrices=design_matrix)

    contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    z_map = glm.compute_contrast(contrast, output_type='z_score')

    return z_map


# Main
if __name__ == '__main__':

    # python3 seed_based_analysis.py --results-dir z_maps --bids-root-dir /media/veracrypt1/synchropioid/fmri_nifti_dir --pet-seed-file /media/veracrypt1/synchropioid/pet_nifti_dir/th-90%_pet_mask.nii.gz --cpu 2 --verbose 1

    t0_total = time.time()

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--results-dir', type=str, default='z_maps',
                        help='Set the name of the plots directory.')
    parser.add_argument('--bids-root-dir', type=str,
                        default='fmri_nifti_dir',
                        help='Set the name of the Nifti preproc directory.')
    parser.add_argument('--pet-seed-file', type=str,
                        default='th-90%_pet_mask.nii.gz',
                        help='Set the name of PET mask file.')
    parser.add_argument('--cache-dir', type=str,
                        default='__cache__',
                        help='Set the name of the cache directory.')
    parser.add_argument('--cpu', type=int, default=1,
                        help='Set the number of CPU for the decomposition.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level.')
    args = parser.parse_args()

    if args.verbose:
        print(f"Saving z-maps under '{args.results_dir}' directory")

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    template_func_paths = (f"{args.bids_root_dir}/derivatives/sub-*/func/"
                           f"sub-*_task-*rest_run-*_space-MNI152Lin_desc-"
                           f"preproc_bold.nii.gz")
    func_paths = glob(template_func_paths)

    participants_fname = os.path.join(args.bids_root_dir, 'participants.tsv')
    participants_path = os.path.abspath(os.path.normpath(participants_fname))
    participants = pd.read_csv(participants_path, sep='\t')
    sub_tag_convert = dict(zip(participants['participant_id'],
                            participants['DICOM tag']))

    if len(func_paths) == 0:
        raise ValueError(f"No results found under '{args.bids_root_dir}'")

    def single_subject_seed_base_analysis(func_path, pet_seed_file, cache_dir,
                                          results_dir):
        """ Seed base analysis for a single subject. """
        func_fname = os.path.basename(func_path)
        run = int(func_fname.split('run-')[1][0])
        sub = func_fname.split('_')[0]
        group_label = ('temgesic' if sub_tag_convert[sub] in temgesic_group
                       else 'control')
        task_type = func_fname.split('task-')[1].split('_')[0]
        t_r = 0.8 if task_type == 'hbrest' else 2.0

        z_map = seed_base_analysis(func_path, args.pet_seed_file,
                                   t_r, args.cache_dir, verbose=0)

        filename = (f"{sub}_run-{run}_group-{group_label}_"
                    f"tr-{t_r:.1f}_z_map.nii.gz")
        filepath = os.path.join(args.results_dir, filename)

        z_map.to_filename(filepath)

        print(f"Saving plot at '{filepath}'")

    Parallel(n_jobs=args.cpu, verbose=100)(delayed(
                        single_subject_seed_base_analysis)(
                                func_path, args.pet_seed_file,
                                args.cache_dir, args.results_dir)
                            for func_path in func_paths)

    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    print(f"Script runs in {delta_t}")
