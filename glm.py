""" Simple GLM analysis of the synchropioid data.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)
import os
import subprocess  # XXX hack to concatenate the spatial maps in one pdf
import shutil
from datetime import datetime
import bids
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from nilearn import input_data, image, plotting
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel


###############################################################################
# Global functions and variables
TR = 2.0

def join_dirs(l_paths):
    """ Joined all paths together. """
    if len(l_paths) == 1:
        return l_paths
    joined_path = l_paths[0]
    for path in l_paths[1:]:
        joined_path = os.path.join(joined_path, path)

    return joined_path

###############################################################################
# Load data
synchropioid_dataset_path = join_dirs([os.path.expanduser('~'), 'DATA',
                                       'synchropioid_data'])
layout = bids.BIDSLayout(synchropioid_dataset_path, derivatives=True)
print("Find Synchropioid dataset at '{}'".format(layout))

###############################################################################
# Saving management
date = datetime.now()
root_dir = 'results_glm_#{0}{1}{2}{3}{4}{5}'.format(date.year, date.month,
                                                    date.day, date.hour,
                                                    date.minute, date.second)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

shutil.copyfile(__file__, os.path.join(root_dir, __file__))

z_maps_dir = os.path.join(root_dir, 'z_maps')
if not os.path.exists(z_maps_dir):
    os.makedirs(z_maps_dir)

plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

###############################################################################
# Main
for idx_sub in ['02', '03', '04']:

    print("Subject-{0}:".format(idx_sub))

    for idx_run in layout.get_runs():

        print("    run-{0}:".format(idx_run))

        derivative_dir = join_dirs([synchropioid_dataset_path, 'derivatives',
                                'fmriprep', 'sub-{}'.format(idx_sub), 'func'])
        func_file = layout.get(scope='derivatives', subject=idx_sub,
                               suffix='preproc', run=idx_run)[0].filename
        func_file = os.path.join(derivative_dir, func_file)

        n_scans = image.load_img(func_file).shape[-1]

        confounds_file = layout.get(scope='derivatives', subject=idx_sub,
                                   suffix='confounds', run=idx_run)[0].filename
        confounds_file = os.path.join(derivative_dir, confounds_file)

        seed_masker = input_data.NiftiSpheresMasker(
                            [(2, -30, -18)], radius=10, detrend=True,
                            standardize=True, low_pass=0.1, high_pass=0.01,
                            t_r=TR, memory='.cache', memory_level=1)
        seed_time_series = seed_masker.fit_transform(func_file)
        df_confounds = pd.DataFrame.from_csv(confounds_file, sep='\t',
                                             header=0)
        df_confounds.fillna(df_confounds.mean(), inplace=True)
        regr = np.hstack([seed_time_series, df_confounds.values])

        frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)

        regr_names = list(df_confounds.columns)
        regr_names.insert(0, 'seed')

        design_matrix = make_first_level_design_matrix(
                            frametimes, hrf_model='spm', add_regs=regr,
                            add_reg_names=regr_names)

        glm = FirstLevelModel(t_r=TR, slice_time_ref=0.5, noise_model='ar1',
                              standardize=True)
        glm.fit(run_imgs=func_file, design_matrices=design_matrix)

        seed_contrast = np.array([1] + [0]*(design_matrix.shape[1] - 1))
        contrast = dict(seed=seed_contrast)
        z_map = glm.compute_contrast(contrast['seed'], output_type='z_score')

        fname = 'z_map_sub-{0}_run-{1}.nii.gz'.format(idx_sub, idx_run)
        fname = os.path.join(z_maps_dir, fname)
        z_map.to_filename(fname)

        th = 0.4 * np.max(np.abs(z_map.get_data()))
        title = 'Z-map sub-{0} run-{1}'.format(idx_sub, idx_run)
        fname = 'z_map_sub-{0}_run-{1}.pdf'.format(idx_sub, idx_run)
        fname = os.path.join(plots_dir, fname)
        plotting.plot_stat_map(z_map, title=title, colorbar=True, threshold=th,
                               display_mode='x', cut_coords=(0.0,),
                               output_file=fname)

    pattern_files = 'z_map_sub-{0}_run-*.pdf'.format(idx_sub)
    pdf_files = os.path.join(plots_dir, pattern_files)
    final_file = 'z_map_sub-{0}_all_run.pdf'.format(idx_sub)
    pdf_file = os.path.join(plots_dir, final_file)
    subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                    shell=True)
    subprocess.call("rm -f {}".format(pdf_files), shell=True)